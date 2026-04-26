import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
import pytorch3d.transforms as transforms

from collections import defaultdict

from custom_pipeline.custom_headpose_dataset import CustomHeadPoseDataset

from egoego.eval.head_pose_metrics import compute_head_pose_metrics
from egoego.model.head_estimation_transformer import HeadFormer
from egoego.model.head_normal_estimation_transformer import HeadNormalFormer

from trainer_amass_cond_motion_diffusion import get_trainer
from utils.data_utils.process_amass_dataset import determine_floor_height_and_contacts

from kinpoly.scripts.eval_metrics_imu_rec import compute_metrics_for_smpl


def _build_head_opt(opt):
    head_opt = argparse.Namespace()
    head_opt.window = opt.head_window
    head_opt.n_dec_layers = opt.head_n_dec_layers
    head_opt.n_head = opt.head_n_head
    head_opt.d_k = opt.head_d_k
    head_opt.d_v = opt.head_d_v
    head_opt.d_model = opt.head_d_model
    head_opt.dist_scale = opt.head_dist_scale
    head_opt.freeze_of_cnn = opt.freeze_of_cnn
    head_opt.input_of_feats = opt.input_of_feats
    return head_opt


def _build_gravity_opt(opt):
    normal_opt = argparse.Namespace()
    normal_opt.window = opt.normal_window
    normal_opt.n_dec_layers = opt.normal_n_dec_layers
    normal_opt.n_head = opt.normal_n_head
    normal_opt.d_k = opt.normal_d_k
    normal_opt.d_v = opt.normal_d_v
    normal_opt.d_model = opt.normal_d_model
    return normal_opt


def _canon_seq_name(name):
    name = str(name)
    if name.endswith(".npz"):
        return name[:-4]
    return name


def _build_full_body_index(full_body_gt_data):
    by_name = {}
    for _, rec in full_body_gt_data.items():
        if "seq_name" not in rec:
            continue
        raw_name = str(rec["seq_name"])
        by_name[raw_name] = rec
        by_name[_canon_seq_name(raw_name)] = rec
    return by_name


def _prepare_gt_full_body_data(diffusion_trainer, rec, device):
    gt_trans = torch.from_numpy(np.asarray(rec["trans"], dtype=np.float32)).to(device)
    gt_root_orient = torch.from_numpy(np.asarray(rec["root_orient"], dtype=np.float32)).to(device)
    gt_pose_aa = torch.from_numpy(np.asarray(rec["body_pose"], dtype=np.float32)).to(device)

    gt_joint_aa = torch.cat((gt_root_orient, gt_pose_aa), dim=-1).reshape(-1, 22, 3)
    global_jrot, global_jpos = diffusion_trainer.ds.fk_smpl(gt_trans, gt_joint_aa)

    floor_height, _, _ = determine_floor_height_and_contacts(global_jpos.detach().cpu().numpy(), fps=30)
    global_jpos[:, :, 2] -= floor_height

    head_idx = 15
    gt_head_pose = torch.cat((global_jpos[:, head_idx, :], global_jrot[:, head_idx, :]), dim=-1)  # T x 7

    return global_jrot, global_jpos, gt_head_pose


def _compute_stage1_metrics(pred_head_pose, gt_head_pose):
    T = min(pred_head_pose.shape[0], gt_head_pose.shape[0])
    pred_head_pose = pred_head_pose[:T]
    gt_head_pose = gt_head_pose[:T]

    pred_trans = pred_head_pose[:, :3].detach().cpu().numpy()
    gt_trans = gt_head_pose[:, :3].detach().cpu().numpy()

    pred_rot = transforms.quaternion_to_matrix(pred_head_pose[:, 3:]).detach().cpu().numpy()
    gt_rot = transforms.quaternion_to_matrix(gt_head_pose[:, 3:]).detach().cpu().numpy()

    return compute_head_pose_metrics(pred_trans, pred_rot, gt_trans, gt_rot)


def evaluate(opt, device):
    # Data
    stage1_data_file = opt.stage1_data_file or os.path.join(opt.data_root_folder, "custom_stage1", "test_headpose_data.p")
    full_body_gt_path = opt.full_body_gt_data_path or os.path.join(
        opt.data_root_folder, "amass_same_shape_egoego_processed", "test_amass_smplh_motion.p"
    )

    if not os.path.exists(stage1_data_file):
        raise FileNotFoundError(f"Cannot find stage1 custom data file: {stage1_data_file}")
    if not os.path.exists(full_body_gt_path):
        raise FileNotFoundError(f"Cannot find stage2 GT file: {full_body_gt_path}")

    dataset = CustomHeadPoseDataset(stage1_data_file, train=False, window=opt.head_window, for_eval=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=True)

    full_body_gt_data = joblib.load(full_body_gt_path)
    full_body_by_name = _build_full_body_index(full_body_gt_data)

    # Models
    head_opt = _build_head_opt(opt)
    head_model = HeadFormer(head_opt, device).to(device)
    head_ckpt = torch.load(opt.head_weight_path, map_location=device)
    head_model.load_state_dict(head_ckpt["transformer_encoder_state_dict"], strict=False)
    head_model.eval()
    print(f"Loaded HeadNet weight: {opt.head_weight_path}")

    normal_opt = _build_gravity_opt(opt)
    normal_model = HeadNormalFormer(normal_opt, device).to(device)
    normal_ckpt = torch.load(opt.normal_weight_path, map_location=device)
    normal_model.load_state_dict(normal_ckpt["transformer_encoder_state_dict"], strict=False)
    normal_model.eval()
    print(f"Loaded GravityNet weight: {opt.normal_weight_path}")

    if not os.path.exists(opt.diffusion_weight_path):
        raise FileNotFoundError(f"Cannot find diffusion checkpoint: {opt.diffusion_weight_path}")
    diffusion_trainer = get_trainer(opt)
    diffusion_trainer.load_weight_path(opt.diffusion_weight_path)
    print(f"Loaded diffusion weight: {opt.diffusion_weight_path}")

    # Stage1 metrics
    s1_e_head_list = []
    s1_o_head_list = []
    s1_t_head_list = []

    # Stage2 metrics
    e_root_list = []
    o_root_list = []
    t_root_list = []
    e_head_list = []
    o_head_list = []
    t_head_list = []
    mpjpe_list = []
    mpjpe_wo_hand_list = []
    single_jpe_list = []

    pred_accl_list = []
    gt_accl_list = []
    accer_list = []
    pred_fs_list = []
    gt_fs_list = []

    skipped_no_gt = 0

    with torch.no_grad():
        for batch in loader:
            seq_name_raw = batch["seq_name"][0]
            seq_name = _canon_seq_name(seq_name_raw)

            gt_rec = full_body_by_name.get(seq_name, None)
            if gt_rec is None:
                skipped_no_gt += 1
                print(f"[Skip] No full-body GT found for sequence: {seq_name_raw}")
                continue

            # Prepare GT full body + GT head for metrics and stage2 alignment.
            global_jrot, global_jpos, gt_head_pose = _prepare_gt_full_body_data(diffusion_trainer, gt_rec, device)

            # Stage1: HeadNet
            head_output = head_model.forward_for_eval(batch)
            pred_scale = head_output["pred_scale"]

            # Stage1: GravityNet
            normal_input = defaultdict(list)
            normal_input["head_trans"] = batch["ori_slam_trans"].to(device) - batch["ori_slam_trans"].to(device)[:, 0:1, :]
            normal_input["head_rot_mat"] = batch["ori_slam_rot_mat"].to(device)
            normal_input["ori_head_pose"] = batch["head_pose"].to(device)
            normal_input["seq_len"] = torch.tensor(normal_input["head_trans"].shape[1], dtype=torch.float32, device=device)[None]

            normal_output = normal_model.forward_for_eval(normal_input, pred_scale=pred_scale)
            pred_head_pose = normal_output["head_pose"]  # 1 x T x 7

            # Match eval_egoego behavior: align XY start and floor.
            pred_head_pose = pred_head_pose.clone()
            pred_head_pose[0, :, :2] -= pred_head_pose[0, 0:1, :2].clone()
            move_to_floor = global_jpos[0:1, 15, :].clone() - pred_head_pose[0, 0:1, :3]
            pred_head_pose[0, :, :3] += move_to_floor

            # Stage1 metrics
            s1_e_head, s1_o_head, s1_t_head = _compute_stage1_metrics(pred_head_pose[0], gt_head_pose)
            s1_e_head_list.append(s1_e_head)
            s1_o_head_list.append(s1_o_head)
            s1_t_head_list.append(s1_t_head)

            # Stage2 generation from predicted head pose.
            pred_local_aa, pred_root_pos = diffusion_trainer.full_body_gen_cond_head_pose_sliding_window(pred_head_pose, seq_name)

            pred_fk_jrot, pred_fk_jpos = diffusion_trainer.ds.fk_smpl(
                pred_root_pos.reshape(-1, 3),
                pred_local_aa.reshape(-1, 22, 3),
            )
            sample_bs = pred_local_aa.shape[0]
            pred_fk_jrot = pred_fk_jrot.reshape(sample_bs, -1, 22, 4)
            pred_fk_jpos = pred_fk_jpos.reshape(sample_bs, -1, 22, 3)

            # Align like official eval.
            gt_move_trans = global_jpos[0:1, 15:16, :].clone()[None].repeat(sample_bs, 1, 1, 1)
            pred_move_trans = pred_fk_jpos[:, 0:1, 15:16, :].clone()
            gt_move_trans[:, :, :, 2] = 0
            pred_move_trans[:, :, :, 2] = 0

            rep_global_jpos = global_jpos[None].repeat(sample_bs, 1, 1, 1) - gt_move_trans
            pred_fk_jpos = pred_fk_jpos - pred_move_trans

            # Custom eval uses one sample.
            pred_floor_height, _, _ = determine_floor_height_and_contacts(pred_fk_jpos[0].detach().cpu().numpy(), fps=30)
            metric_dict = compute_metrics_for_smpl(
                global_jrot[: pred_fk_jrot.shape[1]],
                rep_global_jpos[0, : pred_fk_jpos.shape[1]],
                0.0,
                pred_fk_jrot[0],
                pred_fk_jpos[0],
                pred_floor_height,
            )

            e_root_list.append(metric_dict["root_dist"])
            o_root_list.append(metric_dict["root_rot_dist"])
            t_root_list.append(metric_dict["root_trans_dist"])
            e_head_list.append(metric_dict["head_dist"])
            o_head_list.append(metric_dict["head_rot_dist"])
            t_head_list.append(metric_dict["head_trans_dist"])
            mpjpe_list.append(metric_dict["mpjpe"])
            mpjpe_wo_hand_list.append(metric_dict["mpjpe_wo_hand"])
            single_jpe_list.append(metric_dict["single_jpe"])
            pred_accl_list.append(metric_dict["accel_pred"])
            gt_accl_list.append(metric_dict["accel_gt"])
            accer_list.append(metric_dict["accel_err"])
            pred_fs_list.append(metric_dict["pred_fs"])
            gt_fs_list.append(metric_dict["gt_fs"])

            print(
                f"[{seq_name}] "
                f"S1(T_head={s1_t_head:.2f}mm) | "
                f"S2(MPJPE={metric_dict['mpjpe']:.3f}, MPJPE_wo_hand={metric_dict['mpjpe_wo_hand']:.3f})"
            )

    if len(e_root_list) == 0:
        raise RuntimeError(
            "No sequence evaluated. "
            "Check that stage1 test split and stage2 GT split share sequence names."
        )

    res = {
        "num_sequences": len(e_root_list),
        "num_skipped_no_gt": skipped_no_gt,
        "stage1": {
            "mean_e_head": float(np.mean(np.asarray(s1_e_head_list))),
            "mean_o_head": float(np.mean(np.asarray(s1_o_head_list))),
            "mean_t_head_mm": float(np.mean(np.asarray(s1_t_head_list))),
        },
        "stage2": {
            "mean_e_root": float(np.mean(np.asarray(e_root_list))),
            "mean_o_root": float(np.mean(np.asarray(o_root_list))),
            "mean_t_root": float(np.mean(np.asarray(t_root_list))),
            "mean_e_head": float(np.mean(np.asarray(e_head_list))),
            "mean_o_head": float(np.mean(np.asarray(o_head_list))),
            "mean_t_head": float(np.mean(np.asarray(t_head_list))),
            "mpjpe": float(np.mean(np.asarray(mpjpe_list))),
            "mpjpe_wo_hand": float(np.mean(np.asarray(mpjpe_wo_hand_list))),
            "accel_pred": float(np.mean(np.asarray(pred_accl_list))),
            "accel_gt": float(np.mean(np.asarray(gt_accl_list))),
            "accel_err": float(np.mean(np.asarray(accer_list))),
            "foot_sliding_pred": float(np.mean(np.asarray(pred_fs_list))),
            "foot_sliding_gt": float(np.mean(np.asarray(gt_fs_list))),
        },
    }

    print("==== Custom End-to-End Eval ====")
    print(json.dumps(res, indent=2))

    if opt.output_json:
        out_path = Path(opt.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved overall eval json to: {out_path}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_folder", default="data")
    parser.add_argument("--stage1_data_file", type=str, default="")
    parser.add_argument("--full_body_gt_data_path", type=str, default="")

    parser.add_argument("--head_weight_path", type=str, required=True)
    parser.add_argument("--normal_weight_path", type=str, required=True)
    parser.add_argument("--diffusion_weight_path", type=str, required=True)

    parser.add_argument("--output_json", type=str, default="outputs/egoego_custom_eval.json")

    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")

    # HeadNet settings
    parser.add_argument("--head_window", type=int, default=90)
    parser.add_argument("--head_dist_scale", type=float, default=10.0)
    parser.add_argument("--input_of_feats", action="store_true")
    parser.add_argument("--freeze_of_cnn", action="store_true")
    parser.add_argument("--head_n_dec_layers", type=int, default=2)
    parser.add_argument("--head_n_head", type=int, default=4)
    parser.add_argument("--head_d_k", type=int, default=256)
    parser.add_argument("--head_d_v", type=int, default=256)
    parser.add_argument("--head_d_model", type=int, default=256)

    # GravityNet settings
    parser.add_argument("--normal_window", type=int, default=120)
    parser.add_argument("--normal_n_dec_layers", type=int, default=2)
    parser.add_argument("--normal_n_head", type=int, default=4)
    parser.add_argument("--normal_d_k", type=int, default=256)
    parser.add_argument("--normal_d_v", type=int, default=256)
    parser.add_argument("--normal_d_model", type=int, default=256)

    # Diffusion settings (get_trainer)
    parser.add_argument("--diffusion_window", type=int, default=120)
    parser.add_argument("--diffusion_batch_size", type=int, default=32)
    parser.add_argument("--diffusion_learning_rate", type=float, default=2e-4)
    parser.add_argument("--diffusion_n_dec_layers", type=int, default=4)
    parser.add_argument("--diffusion_n_head", type=int, default=4)
    parser.add_argument("--diffusion_d_k", type=int, default=256)
    parser.add_argument("--diffusion_d_v", type=int, default=256)
    parser.add_argument("--diffusion_d_model", type=int, default=512)
    parser.add_argument("--diffusion_project", type=str, default="exp/stage2_motion_diffusion_custom_runs/train")
    parser.add_argument("--diffusion_exp_name", type=str, default="stage2_cond_motion_diffusion_custom_set1")
    parser.add_argument("--use_min_max", action="store_true")
    parser.add_argument("--canonicalize_init_head", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    opt.diffusion_save_dir = str(Path(opt.diffusion_project) / opt.diffusion_exp_name)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    evaluate(opt, device)
