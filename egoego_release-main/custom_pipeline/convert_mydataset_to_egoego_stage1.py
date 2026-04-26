import argparse
import glob
import os
import re
import struct

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


_FLO_MAGIC = 202021.25


def _find_sequence_dirs(input_root):
    # Case 1: input root itself is one sequence folder.
    if os.path.isdir(os.path.join(input_root, "cam")):
        return [input_root]

    # Case 2: child folders are sequence folders.
    seq_dirs = []
    for name in sorted(os.listdir(input_root)):
        cand = os.path.join(input_root, name)
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "cam")):
            seq_dirs.append(cand)
    return seq_dirs


def _mat3_to_quat_wxyz(rot_mat):
    quat_xyzw = sRot.from_matrix(rot_mat).as_quat()  # x y z w
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def _quat_wxyz_to_mat3(quat_wxyz):
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
    return sRot.from_quat(quat_xyzw).as_matrix().astype(np.float32)


def _compute_head_vels(head_pose, fps=30.0):
    # head_pose: T x 7 [xyz + quat_wxyz]
    dt = 1.0 / fps
    trans = head_pose[:, :3]
    quat = head_pose[:, 3:]

    T = head_pose.shape[0]
    if T < 2:
        return np.zeros((T, 6), dtype=np.float32)

    lin = (trans[1:] - trans[:-1]) / dt  # (T-1) x 3
    ang = []
    for i in range(T - 1):
        r0 = _quat_wxyz_to_mat3(quat[i])
        r1 = _quat_wxyz_to_mat3(quat[i + 1])
        r_rel = r1 @ r0.T
        rotvec = sRot.from_matrix(r_rel).as_rotvec().astype(np.float32)
        ang.append(rotvec / dt)
    ang = np.stack(ang, axis=0)  # (T-1) x 3

    vel = np.concatenate((lin, ang), axis=1)  # (T-1) x 6
    vel = np.concatenate((vel, vel[-1:]), axis=0)  # T x 6
    return vel.astype(np.float32)


def _parse_index_from_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    # Typical names: 000000 / 000000_000001
    if "_" in stem:
        tok = stem.split("_")[0]
    else:
        tok = stem

    m = re.search(r"(\d+)$", tok)
    if not m:
        raise ValueError(f"Cannot parse frame index from: {path}")
    return int(m.group(1))


def _read_flo(path):
    with open(path, "rb") as f:
        magic = struct.unpack("f", f.read(4))[0]
        if abs(magic - _FLO_MAGIC) > 1e-4:
            raise ValueError(f"Invalid .flo magic in {path}: {magic}")
        w = struct.unpack("i", f.read(4))[0]
        h = struct.unpack("i", f.read(4))[0]
        data = np.fromfile(f, np.float32)
    if data.size != h * w * 2:
        raise ValueError(f"Corrupt .flo file: {path}, got {data.size}, expected {h*w*2}")
    return data.reshape(h, w, 2).astype(np.float32)


def _resize_flow(flow, target_hw):
    h, w = target_hw
    if flow.shape[0] == h and flow.shape[1] == w:
        return flow

    if cv2 is None:
        raise RuntimeError(
            "cv2 is required for resizing flow maps when --use_of_feats is not set. "
            "Install opencv-python or provide flow/features already at target size."
        )

    src_h, src_w = flow.shape[:2]
    resized = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    # Scale flow vectors after resize.
    resized[..., 0] *= float(w) / float(src_w)
    resized[..., 1] *= float(h) / float(src_h)
    return resized.astype(np.float32)


def _load_cam_head_pose(seq_dir):
    cam_files = sorted(glob.glob(os.path.join(seq_dir, "cam", "*.npz")))
    if not cam_files:
        return None, None

    frame_idxs = []
    poses = []
    for cam_path in cam_files:
        cam_npz = np.load(cam_path, allow_pickle=True)
        if "pose" not in cam_npz:
            raise KeyError(f"{cam_path} missing key 'pose'")

        pose = np.asarray(cam_npz["pose"], dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected 4x4 pose in {cam_path}, got {pose.shape}")

        rot = pose[:3, :3]
        trans = pose[:3, 3]
        quat_wxyz = _mat3_to_quat_wxyz(rot)

        frame_idxs.append(_parse_index_from_name(cam_path))
        poses.append(np.concatenate((trans, quat_wxyz), axis=0))

    return frame_idxs, np.stack(poses, axis=0).astype(np.float32)


def _find_flow_dir(flow_root, seq_name):
    if not flow_root:
        return None

    candidates = [
        os.path.join(flow_root, seq_name),
        flow_root,
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return None


def _load_flow_inputs(flow_root, seq_name, flow_ext, use_of_feats, flow_map_size):
    flow_dir = _find_flow_dir(flow_root, seq_name)
    if flow_dir is None:
        return None

    flow_paths = sorted(glob.glob(os.path.join(flow_dir, f"*.{flow_ext}")))
    if not flow_paths:
        return None

    flow_idxs = []
    flow_data = []
    for flow_path in flow_paths:
        idx = _parse_index_from_name(flow_path)

        if use_of_feats:
            feat = np.load(flow_path)
            feat = np.asarray(feat, dtype=np.float32).reshape(-1)
            flow_data.append(feat)
        else:
            if flow_ext.lower() == "flo":
                flow = _read_flo(flow_path)
            else:
                flow = np.load(flow_path)
                flow = np.asarray(flow, dtype=np.float32)
                if flow.ndim == 3 and flow.shape[0] == 2 and flow.shape[-1] != 2:
                    flow = np.transpose(flow, (1, 2, 0))
            if flow.ndim != 3 or flow.shape[-1] != 2:
                raise ValueError(f"Expected flow map [H,W,2] in {flow_path}, got {flow.shape}")

            flow = _resize_flow(flow, (flow_map_size, flow_map_size))
            flow_data.append(flow)

        flow_idxs.append(idx)

    # Sort by frame index.
    order = np.argsort(np.asarray(flow_idxs))
    flow_idxs = [flow_idxs[i] for i in order]
    flow_data = [flow_data[i] for i in order]

    return flow_idxs, flow_data


def _infer_quat_order_auto(is_reconstruction_pth):
    # DROID reconstruction.pth commonly stores [tx,ty,tz,qx,qy,qz,qw].
    return "xyzw" if is_reconstruction_pth else "wxyz"


def _quat_to_wxyz(quat, quat_order):
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion must have last dim 4, got {quat.shape}")

    if quat_order == "wxyz":
        return quat.astype(np.float32)
    if quat_order == "xyzw":
        return np.stack((quat[..., 3], quat[..., 0], quat[..., 1], quat[..., 2]), axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported quat order: {quat_order}")


def _resolve_slam_path(slam_path, slam_root, seq_name):
    if slam_path:
        if "{seq_name}" in slam_path:
            resolved = slam_path.format(seq_name=seq_name)
        else:
            resolved = slam_path
        if os.path.exists(resolved):
            return resolved

    if slam_root:
        candidates = [
            os.path.join(slam_root, f"{seq_name}.npy"),
            os.path.join(slam_root, f"{seq_name}.pth"),
            os.path.join(slam_root, seq_name, "reconstruction.pth"),
            os.path.join(slam_root, "reconstruction.pth"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand

    return None


def _load_slam_pose(slam_file, slam_quat_order):
    if slam_file is None:
        return None

    ext = os.path.splitext(slam_file)[1].lower()
    is_pth = ext in {".pth", ".pt"}
    quat_order = slam_quat_order
    if quat_order == "auto":
        quat_order = _infer_quat_order_auto(is_reconstruction_pth=is_pth)

    if is_pth:
        import torch

        payload = torch.load(slam_file, map_location="cpu")
        if not isinstance(payload, dict) or "poses" not in payload:
            raise RuntimeError(f"Unsupported slam .pth format in {slam_file}. Expected dict with 'poses'.")
        poses = payload["poses"]
        if hasattr(poses, "detach"):
            poses = poses.detach().cpu().numpy()
        poses = np.asarray(poses)
    else:
        poses = np.load(slam_file, allow_pickle=True)

    poses = np.asarray(poses)

    if poses.ndim == 2 and poses.shape[1] == 7:
        trans = poses[:, :3].astype(np.float32)
        quat = _quat_to_wxyz(poses[:, 3:], quat_order)
    elif poses.ndim == 3 and poses.shape[1:] == (4, 4):
        trans = poses[:, :3, 3].astype(np.float32)
        quat = np.stack([_mat3_to_quat_wxyz(p[:3, :3]) for p in poses], axis=0).astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported slam pose shape in {slam_file}: {poses.shape}")

    return {
        "trans": trans,
        "quat_wxyz": quat,
    }


def _align_slam_to_gt_first(slam_trans, slam_quat_wxyz, gt_head_pose):
    # Same strategy as official dataset loaders: align first frame rotation/translation to GT.
    gt_trans = gt_head_pose[:, :3]
    gt_quat = gt_head_pose[:, 3:]

    n = min(slam_trans.shape[0], gt_head_pose.shape[0])
    if n < 2:
        return None

    slam_trans = slam_trans[:n]
    slam_quat_wxyz = slam_quat_wxyz[:n]
    gt_trans = gt_trans[:n]
    gt_quat = gt_quat[:n]

    slam_rot = sRot.from_quat(np.stack((slam_quat_wxyz[:, 1], slam_quat_wxyz[:, 2], slam_quat_wxyz[:, 3], slam_quat_wxyz[:, 0]), axis=1)).as_matrix().astype(np.float32)
    gt_rot = sRot.from_quat(np.stack((gt_quat[:, 1], gt_quat[:, 2], gt_quat[:, 3], gt_quat[:, 0]), axis=1)).as_matrix().astype(np.float32)

    pred2gt_rot = gt_rot[0] @ slam_rot[0].T  # 3x3

    aligned_rot = np.matmul(pred2gt_rot[None], slam_rot)
    aligned_trans = np.matmul(pred2gt_rot[None], slam_trans[:, :, None])[:, :, 0]

    move = gt_trans[0:1] - aligned_trans[0:1]
    aligned_trans = aligned_trans + move

    aligned_quat_xyzw = sRot.from_matrix(aligned_rot).as_quat().astype(np.float32)
    aligned_quat_wxyz = np.stack(
        (aligned_quat_xyzw[:, 3], aligned_quat_xyzw[:, 0], aligned_quat_xyzw[:, 1], aligned_quat_xyzw[:, 2]), axis=1
    ).astype(np.float32)

    return {
        "aligned_trans": aligned_trans.astype(np.float32),
        "aligned_quat_wxyz": aligned_quat_wxyz,
        "ori_trans": slam_trans.astype(np.float32),
        "ori_quat_wxyz": slam_quat_wxyz.astype(np.float32),
    }


def _load_sequence(seq_dir, args):
    seq_name = os.path.basename(seq_dir.rstrip("/"))

    cam_idxs, cam_head_pose = _load_cam_head_pose(seq_dir)
    if cam_head_pose is None:
        return None

    flow_bundle = _load_flow_inputs(
        flow_root=args.flow_root,
        seq_name=seq_name,
        flow_ext=args.flow_ext,
        use_of_feats=args.use_of_feats,
        flow_map_size=args.flow_map_size,
    )
    if flow_bundle is None:
        print(f"Skip {seq_name}: no flow files found")
        return None
    _, flow_data = flow_bundle

    slam_file = _resolve_slam_path(args.slam_path, args.slam_root, seq_name)
    slam_pose = _load_slam_pose(slam_file, args.slam_quat_order) if slam_file else None

    # Trim lengths so all stage1 inputs are consistent.
    n_pose = min(cam_head_pose.shape[0], len(flow_data) + 1)
    if slam_pose is not None:
        n_pose = min(n_pose, slam_pose["trans"].shape[0])

    if n_pose < args.min_frames:
        print(f"Skip {seq_name}: too short after alignment ({n_pose} frames)")
        return None

    head_pose = cam_head_pose[:n_pose]
    head_vels = _compute_head_vels(head_pose, fps=args.fps)

    flow_data = flow_data[: n_pose - 1]
    if args.use_of_feats:
        of_feats = np.stack(flow_data, axis=0).astype(np.float32)  # (T-1) x D
        of_data = None
    else:
        of_data = np.stack(flow_data, axis=0).astype(np.float32)  # (T-1) x H x W x 2
        of_feats = None

    quat = head_pose[:, 3:]
    trans = head_pose[:, :3]

    if slam_pose is not None:
        slam_aligned = _align_slam_to_gt_first(slam_pose["trans"], slam_pose["quat_wxyz"], head_pose)
    else:
        slam_aligned = None

    if slam_aligned is None:
        aligned_slam_trans = trans.copy()
        aligned_slam_quat = quat.copy()
        ori_slam_trans = trans.copy()
        ori_slam_quat = quat.copy()
    else:
        aligned_slam_trans = slam_aligned["aligned_trans"]
        aligned_slam_quat = slam_aligned["aligned_quat_wxyz"]
        ori_slam_trans = slam_aligned["ori_trans"]
        ori_slam_quat = slam_aligned["ori_quat_wxyz"]

    rec = {
        "seq_name": seq_name,
        "head_pose": head_pose,  # T x 7
        "head_vels": head_vels,  # T x 6
        # For compatibility with HeadFormer.forward_for_eval().
        "aligned_slam_trans": aligned_slam_trans,
        "aligned_slam_rot_quat": aligned_slam_quat,
        "ori_slam_trans": ori_slam_trans,
        "ori_slam_rot_quat": ori_slam_quat,
    }

    if of_feats is not None:
        rec["of_feats"] = of_feats
    else:
        rec["of"] = of_data

    return rec


def _split_records(records, train_ratio, seed):
    n = len(records)
    if n == 0:
        return [], []
    if n == 1:
        return records, records

    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    split_idx = int(round(n * train_ratio))
    split_idx = max(1, min(split_idx, n - 1))
    train_records = [records[i] for i in order[:split_idx]]
    test_records = [records[i] for i in order[split_idx:]]
    return train_records, test_records


def _as_indexed_dict(records):
    return {idx: rec for idx, rec in enumerate(records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="mydataset", help="Folder containing sequence folders with cam/*.npz")
    parser.add_argument("--output_root", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min_frames", type=int, default=32, help="Minimum number of camera frames to keep a sequence.")
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument(
        "--flow_root",
        type=str,
        default="flow_out",
        help="Flow folder. Supports flow_root/<seq_name>/* or a flat folder."
    )
    parser.add_argument(
        "--flow_ext",
        type=str,
        default="flo",
        choices=["flo", "npy"],
        help="Flow file extension under --flow_root.",
    )
    parser.add_argument(
        "--use_of_feats",
        action="store_true",
        help="Read pre-extracted optical-flow features (1D vectors) instead of raw flow maps.",
    )
    parser.add_argument(
        "--flow_map_size",
        type=int,
        default=224,
        help="Target H/W for flow maps when --use_of_feats is not set.",
    )

    parser.add_argument(
        "--slam_path",
        type=str,
        default="",
        help="Optional slam file path (.npy/.pth). Can include '{seq_name}' template.",
    )
    parser.add_argument(
        "--slam_root",
        type=str,
        default="output_dir",
        help="Optional slam folder. Used when --slam_path is empty.",
    )
    parser.add_argument(
        "--slam_quat_order",
        type=str,
        default="auto",
        choices=["auto", "wxyz", "xyzw"],
        help="Quaternion order in slam file for 7D poses.",
    )

    args = parser.parse_args()

    seq_dirs = _find_sequence_dirs(args.input_root)
    if not seq_dirs:
        raise RuntimeError(
            f"No valid sequences under {args.input_root}. "
            "Expected input_root/cam or input_root/<seq>/cam."
        )

    all_records = []
    for seq_dir in seq_dirs:
        rec = _load_sequence(seq_dir, args)
        if rec is None:
            continue

        # Need at least one flow step.
        num_steps = rec["of_feats"].shape[0] if "of_feats" in rec else rec["of"].shape[0]
        if num_steps < 1:
            print(f"Skip {rec['seq_name']}: empty flow input")
            continue

        all_records.append(rec)

    if not all_records:
        raise RuntimeError("No valid sequence converted.")

    train_records, test_records = _split_records(all_records, args.train_ratio, args.seed)

    out_folder = os.path.join(args.output_root, "custom_stage1")
    os.makedirs(out_folder, exist_ok=True)

    train_path = os.path.join(out_folder, "train_headpose_data.p")
    test_path = os.path.join(out_folder, "test_headpose_data.p")
    all_path = os.path.join(out_folder, "all_headpose_data.p")

    joblib.dump(_as_indexed_dict(train_records), train_path)
    joblib.dump(_as_indexed_dict(test_records), test_path)
    joblib.dump(_as_indexed_dict(all_records), all_path)

    print(f"Converted sequences: {len(all_records)}")
    print(f"Train sequences: {len(train_records)} -> {train_path}")
    print(f"Test sequences: {len(test_records)} -> {test_path}")
    print(f"All sequences: {len(all_records)} -> {all_path}")

    for rec in all_records:
        of_shape = rec["of_feats"].shape if "of_feats" in rec else rec["of"].shape
        print(
            f"  {rec['seq_name']}: head_pose={rec['head_pose'].shape}, "
            f"head_vels={rec['head_vels'].shape}, of={of_shape}, slam={rec['ori_slam_trans'].shape}"
        )


if __name__ == "__main__":
    main()
