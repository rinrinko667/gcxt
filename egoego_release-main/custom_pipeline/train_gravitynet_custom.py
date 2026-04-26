import argparse
import os
from pathlib import Path

import torch
from torch.optim import AdamW
import yaml

try:
    import wandb
except Exception:
    wandb = None

from custom_pipeline.custom_headpose_dataset import CustomGravityDataset
from egoego.model.head_normal_estimation_transformer import HeadNormalFormer


def train(opt, device):
    save_dir = Path(opt.save_dir)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "opt.yaml", "w") as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    train_data_file = opt.train_data_file or os.path.join(opt.data_root_folder, "custom_stage1", "train_headpose_data.p")
    val_data_file = opt.val_data_file or os.path.join(opt.data_root_folder, "custom_stage1", "test_headpose_data.p")

    train_dataset = CustomGravityDataset(train_data_file, train=True, window=opt.window, for_eval=False)
    val_dataset = CustomGravityDataset(val_data_file, train=False, window=opt.window, for_eval=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=False,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training set is empty. Check converted stage1 data files.")

    use_wandb = opt.use_wandb and wandb is not None
    if use_wandb:
        wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

    model = HeadNormalFormer(opt, device).to(device)
    optim = AdamW(params=model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.3)

    for epoch in range(1, opt.epochs + 1):
        model.train()
        train_total_loss = []
        train_normal_loss = []

        for it, batch in enumerate(train_loader):
            output = model(batch)
            total_loss, normal_loss = model.compute_loss(output, batch)

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            optim.step()

            train_total_loss.append(total_loss.detach())
            train_normal_loss.append(normal_loss.detach())

            if it % opt.print_iter == 0:
                print(f"Epoch {epoch} Iter {it} | Total {total_loss.item():.4f} Normal {normal_loss.item():.4f}")

        val_total_loss = []
        val_normal_loss = []
        if epoch % opt.validation_iter == 0 and len(val_loader) > 0:
            model.eval()
            with torch.no_grad():
                for v_it, batch in enumerate(val_loader):
                    if v_it >= opt.max_val_steps:
                        break
                    output = model(batch)
                    total_loss, normal_loss = model.compute_loss(output, batch)
                    val_total_loss.append(total_loss.detach())
                    val_normal_loss.append(normal_loss.detach())

        log_dict = {
            "Train/Loss/Total": torch.stack(train_total_loss).mean().item(),
            "Train/Loss/Normal": torch.stack(train_normal_loss).mean().item(),
        }
        if len(val_total_loss) > 0:
            log_dict.update(
                {
                    "Val/Loss/Total": torch.stack(val_total_loss).mean().item(),
                    "Val/Loss/Normal": torch.stack(val_normal_loss).mean().item(),
                }
            )
            print(
                f"[Val] Epoch {epoch} | "
                f"Total {log_dict['Val/Loss/Total']:.4f} Normal {log_dict['Val/Loss/Normal']:.4f}"
            )

        if use_wandb:
            wandb.log(log_dict)

        scheduler.step()

        if epoch % opt.save_interval == 0:
            ckpt = {
                "epoch": epoch,
                "transformer_encoder_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": torch.stack(train_total_loss).mean().item(),
            }
            ckpt_path = wdir / f"train-{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"[MODEL SAVED] {ckpt_path}")

    if use_wandb:
        wandb.run.finish()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="exp/stage1_gravitynet_custom_runs/train")
    parser.add_argument("--exp_name", default="stage1_gravitynet_custom_set1")
    parser.add_argument("--wandb_pj_name", type=str, default="stage1_gravitynet_custom")
    parser.add_argument("--entity", default="")
    parser.add_argument("--use_wandb", action="store_true")

    parser.add_argument("--data_root_folder", default="data")
    parser.add_argument("--train_data_file", type=str, default="")
    parser.add_argument("--val_data_file", type=str, default="")

    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--validation_iter", type=int, default=1)
    parser.add_argument("--max_val_steps", type=int, default=100)
    parser.add_argument("--print_iter", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--window", type=int, default=120)

    parser.add_argument("--n_dec_layers", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_k", type=int, default=256)
    parser.add_argument("--d_v", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(Path(opt.project) / opt.exp_name)
    opt.exp_name = opt.save_dir.split("/")[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    train(opt, device)
