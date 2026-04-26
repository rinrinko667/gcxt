import random

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms


class CustomHeadPoseDataset(Dataset):
    def __init__(self, data_file, train=True, window=90, for_eval=False):
        self.data_file = data_file
        self.train = train
        self.window = window
        self.for_eval = for_eval

        self.data_dict = joblib.load(self.data_file)
        if not isinstance(self.data_dict, dict):
            raise RuntimeError(f"Expected dict in {self.data_file}, got {type(self.data_dict)}")

        print(f"Loaded {len(self.data_dict)} sequences from {self.data_file}")

    def __len__(self):
        return len(self.data_dict)

    def _slice_window(self, seq_len):
        # seq_len is defined on velocity/of timelines (T), while head_pose is (T+1).
        if self.for_eval:
            start = 0
            end = seq_len
            return start, end

        if seq_len <= self.window:
            return 0, seq_len

        start = random.randint(0, seq_len - self.window)
        end = start + self.window
        return start, end

    def __getitem__(self, index):
        item = self.data_dict[index]
        seq_name = item["seq_name"]

        head_pose = np.asarray(item["head_pose"], dtype=np.float32)  # (T+1) x 7
        head_vels = np.asarray(item["head_vels"], dtype=np.float32)  # (T+1) x 6 or T x 6

        if "of_feats" in item:
            of_data = np.asarray(item["of_feats"], dtype=np.float32)  # T x D
        elif "of" in item:
            of_data = np.asarray(item["of"], dtype=np.float32)  # T x H x W x 2
        else:
            raise KeyError("Stage1 sample missing 'of_feats' or 'of'")

        if head_pose.shape[0] >= 2 and head_vels.shape[0] == head_pose.shape[0]:
            head_vels = head_vels[:-1]

        seq_len = min(head_pose.shape[0] - 1, head_vels.shape[0], of_data.shape[0])
        start, end = self._slice_window(seq_len)

        window_head_pose = head_pose[start : end + 1]
        window_head_vels = head_vels[start:end]
        window_of = of_data[start:end]

        actual_seq_len = window_head_vels.shape[0]

        query = {
            "head_pose": window_head_pose,
            "head_vels": window_head_vels,
            "of": window_of,
            "seq_name": seq_name,
            "seq_len": actual_seq_len,
        }

        # Optional SLAM fields used by forward_for_eval().
        if "aligned_slam_trans" in item and "aligned_slam_rot_quat" in item:
            aligned_slam_trans = np.asarray(item["aligned_slam_trans"], dtype=np.float32)[start : end + 1]
            aligned_slam_rot_quat = np.asarray(item["aligned_slam_rot_quat"], dtype=np.float32)[start : end + 1]
            aligned_slam_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(aligned_slam_rot_quat)).numpy()

            ori_slam_trans = np.asarray(item["ori_slam_trans"], dtype=np.float32)[start : end + 1]
            ori_slam_rot_quat = np.asarray(item["ori_slam_rot_quat"], dtype=np.float32)[start : end + 1]
            ori_slam_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(ori_slam_rot_quat)).numpy()

            query["aligned_slam_trans"] = aligned_slam_trans
            query["aligned_slam_rot_quat"] = aligned_slam_rot_quat
            query["aligned_slam_rot_mat"] = aligned_slam_rot_mat

            query["ori_slam_trans"] = ori_slam_trans
            query["ori_slam_rot_quat"] = ori_slam_rot_quat
            query["ori_slam_rot_mat"] = ori_slam_rot_mat
        else:
            # Fallback: use GT head pose as SLAM input.
            fallback_trans = window_head_pose[:, :3]
            fallback_quat = window_head_pose[:, 3:]
            fallback_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(fallback_quat)).numpy()
            query["aligned_slam_trans"] = fallback_trans
            query["aligned_slam_rot_quat"] = fallback_quat
            query["aligned_slam_rot_mat"] = fallback_rot_mat

            query["ori_slam_trans"] = fallback_trans
            query["ori_slam_rot_quat"] = fallback_quat
            query["ori_slam_rot_mat"] = fallback_rot_mat

        return query


class CustomGravityDataset(Dataset):
    def __init__(self, data_file, train=True, window=120, for_eval=False):
        self.data_file = data_file
        self.train = train
        self.window = window
        self.for_eval = for_eval

        self.data_dict = joblib.load(self.data_file)
        if not isinstance(self.data_dict, dict):
            raise RuntimeError(f"Expected dict in {self.data_file}, got {type(self.data_dict)}")

        print(f"Loaded {len(self.data_dict)} sequences from {self.data_file}")

    def __len__(self):
        return len(self.data_dict)

    @staticmethod
    def _augment_w_rotation(ori_head_pose):
        ori_head_trans = ori_head_pose[:, :3]  # T x 3
        ori_head_quat = ori_head_pose[:, 3:]  # T x 4
        ori_head_rot_mat = transforms.quaternion_to_matrix(ori_head_quat)  # T x 3 x 3

        random_rot_mat = transforms.random_rotation()[None]  # 1 x 3 x 3
        aug_head_rot_mat = torch.matmul(random_rot_mat, ori_head_rot_mat)  # T x 3 x 3

        centered_head_trans = ori_head_trans - ori_head_trans[0:1]
        aug_head_trans = torch.matmul(random_rot_mat, centered_head_trans[:, :, None])[:, :, 0]  # T x 3

        ori_floor_normal = torch.tensor([0.0, 0.0, 1.0]).float()[:, None]
        aug_floor_normal = torch.matmul(random_rot_mat[0], ori_floor_normal)  # 3 x 1

        return random_rot_mat, aug_head_rot_mat, aug_head_trans, aug_floor_normal

    @staticmethod
    def _augment_w_scale(head_trans):
        random_scale = np.random.uniform(low=0.1, high=10.0, size=(1,)).astype(np.float32)[0]
        head_trans_diff = head_trans[1:] - head_trans[:-1]
        trans_diff_after_scale = head_trans_diff * random_scale

        trans_after_scale = [head_trans[0:1]]
        for t_idx in range(head_trans.shape[0] - 1):
            trans_after_scale.append(trans_after_scale[-1] + trans_diff_after_scale[t_idx : t_idx + 1])

        aug_head_trans = torch.cat(trans_after_scale, dim=0)
        return random_scale, aug_head_trans

    def _slice_window(self, seq_len):
        # seq_len on head_pose timeline (T+1).
        if self.for_eval:
            return 0, min(seq_len, self.window + 1)

        if seq_len <= self.window + 1:
            return 0, seq_len

        start = random.randint(0, seq_len - (self.window + 1))
        end = start + self.window + 1
        return start, end

    def __getitem__(self, index):
        item = self.data_dict[index]
        seq_name = item["seq_name"]
        seq_head_pose = torch.from_numpy(np.asarray(item["head_pose"], dtype=np.float32))

        seq_len = seq_head_pose.shape[0]
        start, end = self._slice_window(seq_len)
        window_head_pose = seq_head_pose[start:end]

        aug_rot_mat, aug_head_rot_mat, aug_head_trans, aug_floor_normal = self._augment_w_rotation(window_head_pose)
        aug_scale, aug_head_trans = self._augment_w_scale(aug_head_trans)

        actual_seq_len = window_head_pose.shape[0]
        if actual_seq_len < self.window + 1:
            pad = self.window + 1 - actual_seq_len
            window_head_pose = torch.cat((window_head_pose, torch.zeros(pad, 7)), dim=0)
            aug_head_rot_mat = torch.cat((aug_head_rot_mat, torch.zeros(pad, 3, 3)), dim=0)
            aug_head_trans = torch.cat((aug_head_trans, torch.zeros(pad, 3)), dim=0)

        query = {
            "ori_head_pose": window_head_pose,  # (window+1) x 7
            "head_rot_mat": aug_head_rot_mat,  # (window+1) x 3 x 3
            "head_trans": aug_head_trans,  # (window+1) x 3
            "seq_len": actual_seq_len,
            "seq_name": seq_name,
            "aligned_rot_mat": aug_rot_mat[0].T,  # 3 x 3
            "aligned_scale": 1.0 / float(aug_scale),
            "floor_normal": aug_floor_normal,  # 3 x 1
        }
        return query
