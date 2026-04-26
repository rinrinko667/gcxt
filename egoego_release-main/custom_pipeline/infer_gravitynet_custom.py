import argparse
import os
from pathlib import Path

import numpy as np
import torch
import pytorch3d.transforms as transforms

from custom_pipeline.custom_headpose_dataset import CustomGravityDataset
from egoego.model.head_normal_estimation_transformer import HeadNormalFormer


def _find_index_by_name(dataset, seq_name):
    for idx in range(len(dataset)):
        if dataset.data_dict[idx]["seq_name"] == seq_name:
            return idx
    raise ValueError(f"Cannot find seq_name={seq_name}")


def _to_batch_tensor(sample, device):
    batch = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            batch[k] = torch.from_numpy(v)[None].to(device)
        elif isinstance(v, (int, float)):
            batch[k] = torch.tensor([v]).to(device)
        elif isinstance(v, str):
            batch[k] = [v]
        else:
            batch[k] = v
    return batch


def _build_batch_from_pose(head_pose, device):
    # head_pose: T x 7 [xyz + quat_wxyz]
    head_pose = np.asarray(head_pose, dtype=np.float32)
    quat = torch.from_numpy(head_pose[:, 3:]).float()
    rot_mat = transforms.quaternion_to_matrix(quat).numpy()

    sample = {
        "seq_name": "external_pose",
        "ori_head_pose": head_pose,
        "head_trans": head_pose[:, :3].copy(),
        "head_rot_mat": rot_mat,
        "seq_len": head_pose.shape[0],
        "aligned_rot_mat": np.eye(3, dtype=np.float32),
        "aligned_scale": 1.0,
        "floor_normal": np.array([[0.0], [0.0], [1.0]], dtype=np.float32),
    }
    return _to_batch_tensor(sample, device), sample["seq_name"]


def run(opt, device):
    model = HeadNormalFormer(opt, device).to(device)
    ckpt = torch.load(opt.weight_path, map_location=device)
    model.load_state_dict(ckpt["transformer_encoder_state_dict"], strict=False)
    model.eval()
    print(f"Loaded GravityNet weight: {opt.weight_path}")

    pred_scale = None
    if opt.pred_scale is not None:
        pred_scale = torch.tensor(float(opt.pred_scale), device=device)

    if opt.headnet_output_npz:
        npz = np.load(opt.headnet_output_npz, allow_pickle=True)
        if "ori_slam_pose" in npz:
            slam_pose = npz["ori_slam_pose"]
        elif "pred_head_pose" in npz:
            slam_pose = npz["pred_head_pose"]
        else:
            raise KeyError(f"{opt.headnet_output_npz} missing ori_slam_pose/pred_head_pose")

        if pred_scale is None and "pred_scale" in npz:
            pred_scale = torch.tensor(float(npz["pred_scale"]), device=device)

        batch, seq_name = _build_batch_from_pose(slam_pose, device)
    elif opt.head_pose_npy:
        head_pose = np.load(opt.head_pose_npy)
        batch, seq_name = _build_batch_from_pose(head_pose, device)
    else:
        data_file = opt.data_file or os.path.join(opt.data_root_folder, "custom_stage1", "test_headpose_data.p")
        dataset = CustomGravityDataset(data_file, train=False, window=opt.window, for_eval=True)
        if opt.seq_name:
            seq_index = _find_index_by_name(dataset, opt.seq_name)
        else:
            seq_index = opt.seq_index
        sample = dataset[seq_index]
        seq_name = sample["seq_name"]
        batch = _to_batch_tensor(sample, device)

    with torch.no_grad():
        output = model.forward_for_eval(batch, pred_scale=pred_scale)

    corrected_head_pose = output["head_pose"][0].detach().cpu().numpy()
    corrected_head_trans = output["head_trans"][0].detach().cpu().numpy()
    corrected_head_rot_mat = output["head_rot_mat"][0].detach().cpu().numpy()

    out_path = Path(opt.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        seq_name=seq_name,
        corrected_head_pose=corrected_head_pose,
        corrected_head_trans=corrected_head_trans,
        corrected_head_rot_mat=corrected_head_rot_mat,
    )
    print(f"Saved GravityNet inference output: {out_path}")
    print(f"Sequence: {seq_name}, corrected_head_pose={corrected_head_pose.shape}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_folder", default="data")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--seq_index", type=int, default=0)
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--output_npz", type=str, default="outputs/gravitynet_custom_infer.npz")

    parser.add_argument(
        "--headnet_output_npz",
        type=str,
        default="",
        help="Optional output from infer_headnet_custom.py, used to provide slam pose and pred_scale.",
    )
    parser.add_argument("--head_pose_npy", type=str, default="", help="Optional external head pose .npy (T x 7).")
    parser.add_argument("--pred_scale", type=float, default=None, help="Optional explicit scale (overrides npz scale).")

    parser.add_argument("--device", default="0")
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--n_dec_layers", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_k", type=int, default=256)
    parser.add_argument("--d_v", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    run(opt, device)
