import argparse
import os
from pathlib import Path

import numpy as np
import torch

from custom_pipeline.custom_headpose_dataset import CustomHeadPoseDataset
from egoego.model.head_estimation_transformer import HeadFormer


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


def run(opt, device):
    data_file = opt.data_file or os.path.join(opt.data_root_folder, "custom_stage1", "test_headpose_data.p")
    dataset = CustomHeadPoseDataset(data_file, train=False, window=opt.window, for_eval=True)

    if opt.seq_name:
        seq_index = _find_index_by_name(dataset, opt.seq_name)
    else:
        seq_index = opt.seq_index
    sample = dataset[seq_index]
    seq_name = sample["seq_name"]

    model = HeadFormer(opt, device).to(device)
    ckpt = torch.load(opt.weight_path, map_location=device)
    model.load_state_dict(ckpt["transformer_encoder_state_dict"], strict=False)
    model.eval()
    print(f"Loaded HeadNet weight: {opt.weight_path}")

    with torch.no_grad():
        batch = _to_batch_tensor(sample, device)
        output = model.forward_for_eval(batch)

    pred_head_pose = output["head_pose"][0].detach().cpu().numpy()
    pred_scale = float(output["pred_scale"].detach().cpu().item())
    gt_head_pose = sample["head_pose"].astype(np.float32)
    ori_slam_pose = np.concatenate((sample["ori_slam_trans"], sample["ori_slam_rot_quat"]), axis=-1).astype(np.float32)

    out_path = Path(opt.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        seq_name=seq_name,
        pred_head_pose=pred_head_pose,
        pred_scale=pred_scale,
        gt_head_pose=gt_head_pose,
        ori_slam_pose=ori_slam_pose,
    )
    print(f"Saved HeadNet inference output: {out_path}")
    print(f"Sequence: {seq_name}, pred_scale={pred_scale:.6f}, pred_head_pose={pred_head_pose.shape}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_folder", default="data")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--seq_index", type=int, default=0)
    parser.add_argument("--seq_name", type=str, default="")
    parser.add_argument("--output_npz", type=str, default="outputs/headnet_custom_infer.npz")

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
    run(opt, device)
