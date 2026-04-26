import argparse
import glob
import os

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot


def _as_axis_angle_root(root_orient):
    root_orient = np.asarray(root_orient)
    if root_orient.shape == (3,):
        return root_orient.astype(np.float32)
    if root_orient.shape == (3, 3):
        return sRot.from_matrix(root_orient).as_rotvec().astype(np.float32)
    if root_orient.shape == (1, 3, 3):
        return sRot.from_matrix(root_orient[0]).as_rotvec().astype(np.float32)
    raise ValueError(f"Unsupported global_orient shape: {root_orient.shape}")


def _as_axis_angle_body(body_pose):
    body_pose = np.asarray(body_pose)
    if body_pose.shape == (63,):
        return body_pose.astype(np.float32)
    if body_pose.shape == (21, 3):
        return body_pose.reshape(-1).astype(np.float32)
    if body_pose.shape == (21, 3, 3):
        return sRot.from_matrix(body_pose).as_rotvec().reshape(-1).astype(np.float32)
    raise ValueError(f"Unsupported body_pose shape: {body_pose.shape}")


def _pad_betas_16(betas):
    betas = np.asarray(betas).reshape(-1).astype(np.float32)
    if betas.shape[0] >= 16:
        return betas[:16]
    out = np.zeros(16, dtype=np.float32)
    out[: betas.shape[0]] = betas
    return out


def _find_sequence_dirs(input_root):
    # Case 1: input root itself is one sequence folder.
    if os.path.isdir(os.path.join(input_root, "smpl")):
        return [input_root]

    # Case 2: child folders are sequence folders.
    seq_dirs = []
    for name in sorted(os.listdir(input_root)):
        cand = os.path.join(input_root, name)
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "smpl")):
            seq_dirs.append(cand)
    return seq_dirs


def _load_sequence(seq_dir, default_gender):
    smpl_dir = os.path.join(seq_dir, "smpl")
    frame_files = sorted(glob.glob(os.path.join(smpl_dir, "*.npz")))
    if not frame_files:
        return None

    root_orient = []
    body_pose = []
    trans = []
    betas = []
    gender = default_gender

    for frame_path in frame_files:
        frame = np.load(frame_path, allow_pickle=True)
        if "global_orient" not in frame or "body_pose" not in frame:
            raise KeyError(f"{frame_path} is missing global_orient/body_pose")

        if "transl" in frame:
            curr_trans = frame["transl"]
        elif "trans" in frame:
            curr_trans = frame["trans"]
        else:
            raise KeyError(f"{frame_path} is missing transl/trans")

        root_orient.append(_as_axis_angle_root(frame["global_orient"]))
        body_pose.append(_as_axis_angle_body(frame["body_pose"]))
        trans.append(np.asarray(curr_trans).reshape(3).astype(np.float32))

        if "betas" in frame:
            betas.append(frame["betas"])
        if "gender" in frame:
            g = frame["gender"]
            if isinstance(g, np.ndarray):
                g = g.item() if g.shape == () else g.tolist()
            if isinstance(g, bytes):
                g = g.decode("utf-8")
            gender = str(g)

    root_orient = np.stack(root_orient, axis=0)  # T x 3
    body_pose = np.stack(body_pose, axis=0)  # T x 63
    trans = np.stack(trans, axis=0)  # T x 3

    if betas:
        betas = _pad_betas_16(np.stack([np.asarray(b).reshape(-1) for b in betas], axis=0).mean(axis=0))
    else:
        betas = np.zeros(16, dtype=np.float32)

    seq_name = os.path.basename(seq_dir.rstrip("/"))
    return {
        "root_orient": root_orient,
        "body_pose": body_pose,
        "trans": trans,
        "beta": betas,
        "seq_name": seq_name,
        "gender": gender,
    }


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
    parser.add_argument(
        "--input_root",
        type=str,
        default="mydataset",
        help="Root containing sequence folders. Each sequence folder should include smpl/*.npz.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data",
        help="Output data root folder. Files will be written to output_root/amass_same_shape_egoego_processed/.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Sequence-level train split ratio.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for split.")
    parser.add_argument(
        "--default_gender",
        type=str,
        default="male",
        choices=["male", "female", "neutral"],
        help="Used when gender is not present in input files.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=31,
        help="Minimum frames for keeping a sequence. Window training needs at least 30 frames.",
    )
    args = parser.parse_args()

    seq_dirs = _find_sequence_dirs(args.input_root)
    if not seq_dirs:
        raise RuntimeError(
            f"No sequence folders found under {args.input_root}. "
            "Expected either input_root/smpl/*.npz or input_root/<seq_name>/smpl/*.npz."
        )

    all_records = []
    for seq_dir in seq_dirs:
        rec = _load_sequence(seq_dir, args.default_gender)
        if rec is None:
            continue
        if rec["trans"].shape[0] < args.min_frames:
            print(f"Skip {rec['seq_name']}: too short ({rec['trans'].shape[0]} frames)")
            continue
        all_records.append(rec)

    if not all_records:
        raise RuntimeError("No valid sequences were converted.")

    train_records, test_records = _split_records(all_records, args.train_ratio, args.seed)

    out_folder = os.path.join(args.output_root, "amass_same_shape_egoego_processed")
    os.makedirs(out_folder, exist_ok=True)

    train_path = os.path.join(out_folder, "train_amass_smplh_motion.p")
    test_path = os.path.join(out_folder, "test_amass_smplh_motion.p")
    all_path = os.path.join(out_folder, "amass_smplh_motion.p")

    joblib.dump(_as_indexed_dict(train_records), train_path)
    joblib.dump(_as_indexed_dict(test_records), test_path)
    joblib.dump(_as_indexed_dict(all_records), all_path)

    print(f"Converted sequences: {len(all_records)}")
    print(f"Train sequences: {len(train_records)} -> {train_path}")
    print(f"Test sequences: {len(test_records)} -> {test_path}")
    print(f"All sequences: {len(all_records)} -> {all_path}")
    print("Per-sequence frame counts:")
    for rec in all_records:
        print(f"  {rec['seq_name']}: {rec['trans'].shape[0]}")


if __name__ == "__main__":
    main()
