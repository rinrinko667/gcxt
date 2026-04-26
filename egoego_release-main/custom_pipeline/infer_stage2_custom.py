import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import torch

from trainer_amass_cond_motion_diffusion import get_trainer
from utils.data_utils.process_amass_dataset import determine_floor_height_and_contacts


def _resolve_default_test_path(data_root_folder):
    return os.path.join(
        data_root_folder, "amass_same_shape_egoego_processed", "test_amass_smplh_motion.p"
    )


def _select_sequence(data_dict, seq_index, seq_name):
    if not isinstance(data_dict, dict) or len(data_dict) == 0:
        raise RuntimeError("Input full-body data file is empty or not a dict.")

    keys = list(data_dict.keys())
    records = [data_dict[k] for k in keys]

    if seq_name:
        for rec in records:
            if str(rec.get("seq_name", "")) == seq_name:
                return rec
        raise ValueError(f"Cannot find seq_name={seq_name} in full-body data.")

    if seq_index < 0 or seq_index >= len(records):
        raise IndexError(f"seq_index={seq_index} is out of range. Total sequences={len(records)}")

    return records[seq_index]


def _head_pose_from_gt_smpl(sample, trainer, device):
    required_keys = ["trans", "root_orient", "body_pose"]
    for key in required_keys:
        if key not in sample:
            raise KeyError(f"Sample is missing key: {key}")

    trans = torch.from_numpy(sample["trans"]).float().to(device)
    root_orient = torch.from_numpy(sample["root_orient"]).float().to(device)
    body_pose = torch.from_numpy(sample["body_pose"]).float().reshape(-1, 21, 3).to(device)
    pose_aa = torch.cat((root_orient[:, None, :], body_pose), dim=1)  # T x 22 x 3

    global_jrot, global_jpos = trainer.ds.fk_smpl(trans, pose_aa)

    # Keep floor convention consistent with eval_stage2.py.
    floor_height, _, _ = determine_floor_height_and_contacts(global_jpos.detach().cpu().numpy(), fps=30)
    global_jpos[:, :, 2] -= floor_height

    head_idx = 15
    head_pose = torch.cat((global_jpos[:, head_idx, :], global_jrot[:, head_idx, :]), dim=-1)  # T x 7
    return head_pose[None]  # 1 x T x 7


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="0", help="CUDA device index.")
    parser.add_argument("--data_root_folder", type=str, default="data", help="Data root for AMASS-style stage2 files.")

    parser.add_argument(
        "--full_body_data_path",
        type=str,
        default="",
        help="Optional explicit path to test_amass_smplh_motion.p. If empty, use data_root_folder default.",
    )
    parser.add_argument("--seq_index", type=int, default=0, help="Sequence index when reading full_body_data_path.")
    parser.add_argument("--seq_name", type=str, default="", help="Optional exact seq_name override.")
    parser.add_argument(
        "--head_pose_npy",
        type=str,
        default="",
        help="Optional .npy with shape [T,7] (xyz + quat_wxyz). If set, bypass GT head extraction.",
    )
    parser.add_argument(
        "--head_pose_npz",
        type=str,
        default="",
        help="Optional .npz containing head pose array. If set, bypass GT head extraction.",
    )
    parser.add_argument(
        "--head_pose_key",
        type=str,
        default="corrected_head_pose",
        help="Key used when loading --head_pose_npz.",
    )

    parser.add_argument("--output_npz", type=str, default="outputs/stage2_infer_result.npz", help="Output npz path.")
    parser.add_argument("--gen_vis", action="store_true", help="Generate mesh rendering by Blender.")
    parser.add_argument("--vis_dir", type=str, default="outputs/stage2_vis", help="Visualization output folder.")

    parser.add_argument(
        "--diffusion_weight_path",
        type=str,
        default="",
        help="Optional explicit diffusion checkpoint. If empty, use ./pretrained_models/stage2_diffusion_4.pt",
    )

    # Diffusion model settings (same as eval_stage2.py/get_trainer()).
    parser.add_argument("--diffusion_window", type=int, default=120, help="Horizon.")
    parser.add_argument("--diffusion_batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--diffusion_learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--diffusion_n_dec_layers", type=int, default=4, help="Transformer decoder layers.")
    parser.add_argument("--diffusion_n_head", type=int, default=4, help="Transformer heads.")
    parser.add_argument("--diffusion_d_k", type=int, default=256, help="Transformer key dim.")
    parser.add_argument("--diffusion_d_v", type=int, default=256, help="Transformer value dim.")
    parser.add_argument("--diffusion_d_model", type=int, default=512, help="Transformer model dim.")
    parser.add_argument("--diffusion_project", type=str, default="runs/train", help="Project path.")
    parser.add_argument("--diffusion_exp_name", type=str, default="", help="Experiment name.")
    parser.add_argument("--canonicalize_init_head", action="store_true")
    parser.add_argument("--use_min_max", action="store_true")

    args = parser.parse_args()

    args.diffusion_save_dir = str(Path(args.diffusion_project) / args.diffusion_exp_name)

    torch_device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")

    trainer = get_trainer(args)
    if args.diffusion_weight_path:
        weight_path = args.diffusion_weight_path
    else:
        weight_path = os.path.join("pretrained_models", "stage2_diffusion_4.pt")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Cannot find diffusion checkpoint: {weight_path}")
    trainer.load_weight_path(weight_path)
    print(f"Loaded diffusion checkpoint: {weight_path}")

    seq_name = "custom_sequence"
    if args.head_pose_npz:
        npz = np.load(args.head_pose_npz, allow_pickle=True)
        if args.head_pose_key not in npz:
            raise KeyError(
                f"Key '{args.head_pose_key}' not found in {args.head_pose_npz}. "
                f"Available keys: {list(npz.keys())}"
            )
        head_pose = npz[args.head_pose_key]
        if head_pose.ndim != 2 or head_pose.shape[1] != 7:
            raise ValueError(f"head_pose in npz must have shape [T,7], got {head_pose.shape}")
        head_pose = torch.from_numpy(head_pose).float().to(torch_device)[None]  # 1 x T x 7
        seq_name = Path(args.head_pose_npz).stem
    elif args.head_pose_npy:
        head_pose = np.load(args.head_pose_npy)
        if head_pose.ndim != 2 or head_pose.shape[1] != 7:
            raise ValueError(f"head_pose_npy must have shape [T,7], got {head_pose.shape}")
        head_pose = torch.from_numpy(head_pose).float().to(torch_device)[None]  # 1 x T x 7
        seq_name = Path(args.head_pose_npy).stem
    else:
        full_body_data_path = args.full_body_data_path or _resolve_default_test_path(args.data_root_folder)
        if not os.path.exists(full_body_data_path):
            raise FileNotFoundError(f"Cannot find full-body data file: {full_body_data_path}")

        full_body_data = joblib.load(full_body_data_path)
        sample = _select_sequence(full_body_data, args.seq_index, args.seq_name)
        seq_name = str(sample.get("seq_name", f"seq_{args.seq_index}"))
        head_pose = _head_pose_from_gt_smpl(sample, trainer, torch_device)
        print(f"Using GT-derived head pose from sequence: {seq_name}")

    with torch.no_grad():
        pred_local_aa, pred_root_trans = trainer.full_body_gen_cond_head_pose_sliding_window(head_pose, seq_name)

        pred_fk_jrot, pred_fk_jpos = trainer.ds.fk_smpl(
            pred_root_trans.reshape(-1, 3), pred_local_aa.reshape(-1, 22, 3)
        )

        batch_size = pred_local_aa.shape[0]
        num_frames = pred_local_aa.shape[1]
        pred_fk_jrot = pred_fk_jrot.reshape(batch_size, num_frames, 22, 4)
        pred_fk_jpos = pred_fk_jpos.reshape(batch_size, num_frames, 22, 3)

    output_npz = args.output_npz
    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
    np.savez(
        output_npz,
        seq_name=seq_name,
        input_head_pose=head_pose[0].detach().cpu().numpy(),
        pred_local_aa=pred_local_aa[0].detach().cpu().numpy(),
        pred_root_trans=pred_root_trans[0].detach().cpu().numpy(),
        pred_global_jquat=pred_fk_jrot[0].detach().cpu().numpy(),
        pred_global_jpos=pred_fk_jpos[0].detach().cpu().numpy(),
    )
    print(f"Saved inference result: {output_npz}")

    if args.gen_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        trainer.gen_full_body_vis(pred_root_trans[0], pred_local_aa[0], args.vis_dir, seq_name)
        print(f"Saved visualization under: {args.vis_dir}")


if __name__ == "__main__":
    main()
