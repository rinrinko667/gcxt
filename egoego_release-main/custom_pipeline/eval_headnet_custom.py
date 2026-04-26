import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import pytorch3d.transforms as transforms

from custom_pipeline.custom_headpose_dataset import CustomHeadPoseDataset
from egoego.eval.head_pose_metrics import compute_head_pose_metrics
from egoego.model.head_estimation_transformer import HeadFormer


def evaluate(opt, device):
    data_file = opt.data_file or os.path.join(opt.data_root_folder, "custom_stage1", "test_headpose_data.p")
    dataset = CustomHeadPoseDataset(data_file, train=False, window=opt.window, for_eval=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=True)

    model = HeadFormer(opt, device).to(device)
    ckpt = torch.load(opt.weight_path, map_location=device)
    model.load_state_dict(ckpt["transformer_encoder_state_dict"], strict=False)
    model.eval()
    print(f"Loaded HeadNet weight: {opt.weight_path}")

    e_head_list = []
    o_head_list = []
    t_head_list = []

    with torch.no_grad():
        for batch in loader:
            seq_name = batch["seq_name"][0]
            output = model.forward_for_eval(batch)

            pred_pose = output["head_pose"][0].detach().cpu()
            gt_pose = batch["head_pose"][0].detach().cpu()

            T = min(pred_pose.shape[0], gt_pose.shape[0])
            pred_pose = pred_pose[:T]
            gt_pose = gt_pose[:T]

            pred_trans = pred_pose[:, :3].numpy()
            gt_trans = gt_pose[:, :3].numpy()

            pred_rot = transforms.quaternion_to_matrix(pred_pose[:, 3:]).numpy()
            gt_rot = transforms.quaternion_to_matrix(gt_pose[:, 3:]).numpy()

            e_head, o_head, t_head = compute_head_pose_metrics(pred_trans, pred_rot, gt_trans, gt_rot)
            e_head_list.append(e_head)
            o_head_list.append(o_head)
            t_head_list.append(t_head)

            print(f"[{seq_name}] E_head={e_head:.6f}, O_head={o_head:.6f}, T_head(mm)={t_head:.3f}")

    if len(e_head_list) == 0:
        raise RuntimeError("No sequence evaluated.")

    res = {
        "num_sequences": len(e_head_list),
        "mean_e_head": float(np.mean(e_head_list)),
        "mean_o_head": float(np.mean(o_head_list)),
        "mean_t_head_mm": float(np.mean(t_head_list)),
    }
    print("==== HeadNet Eval ====")
    print(res)

    if opt.output_json:
        out_path = Path(opt.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved eval json to: {out_path}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_folder", default="data")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="outputs/headnet_custom_eval.json")

    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--window", type=int, default=90)

    parser.add_argument("--dist_scale", type=float, default=10.0)
    parser.add_argument("--freeze_of_cnn", action="store_true")
    parser.add_argument("--input_of_feats", action="store_true")

    parser.add_argument("--n_dec_layers", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_k", type=int, default=256)
    parser.add_argument("--d_v", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    evaluate(opt, device)
