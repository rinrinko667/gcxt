import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def filtered_image_stream(imagedir, calib, stride, t0=0):
    """image generator with t0 filtering"""
    for (t, image, intrinsics) in image_stream(imagedir, calib, stride):
        if t >= t0:
            yield t, image, intrinsics

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, traj_est, imagedir, calib, stride, t0, save_path):

    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    frame_data = list(filtered_image_stream(imagedir, calib, stride, t0))
    if len(frame_data) == 0:
        raise RuntimeError("no frames available for reconstruction saving")

    frame_tstamps = torch.as_tensor([x[0] for x in frame_data], dtype=torch.float32)
    frame_images = torch.stack([x[1][0].cpu() for x in frame_data], dim=0)
    frame_intrinsics = torch.stack([x[2].cpu() / 8.0 for x in frame_data], dim=0)
    frame_poses = torch.from_numpy(traj_est).float()

    if frame_poses.shape[0] != frame_tstamps.shape[0]:
        n = min(frame_poses.shape[0], frame_tstamps.shape[0])
        print(
            f"[warn] pose/frame mismatch ({frame_poses.shape[0]} vs {frame_tstamps.shape[0]}), truncating to {n}"
        )
        frame_tstamps = frame_tstamps[:n]
        frame_images = frame_images[:n]
        frame_intrinsics = frame_intrinsics[:n]
        frame_poses = frame_poses[:n]

    key_t = video.counter.value
    key_tstamps = video.tstamp[:key_t].cpu()
    key_disps = video.disps_up[:key_t].cpu()

    if key_t > 0:
        right = torch.bucketize(frame_tstamps, key_tstamps)
        right = torch.clamp(right, max=key_t - 1)
        left = torch.clamp(right - 1, min=0)

        choose_left = (frame_tstamps - key_tstamps[left]).abs() <= (key_tstamps[right] - frame_tstamps).abs()
        nearest = torch.where(choose_left, left, right)
        frame_disps = key_disps[nearest]
    else:
        h, w = frame_images.shape[-2:]
        frame_disps = torch.ones((frame_images.shape[0], h, w), dtype=torch.float32)

    save_data = {
        "tstamps": frame_tstamps,
        "images": frame_images,
        "disps": frame_disps,
        "poses": frame_poses,
        "intrinsics": frame_intrinsics,
    }

    torch.save(save_data, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    for (t, image, intrinsics) in tqdm(filtered_image_stream(args.imagedir, args.calib, args.stride, args.t0)):

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(filtered_image_stream(args.imagedir, args.calib, args.stride, args.t0))
    
    if args.reconstruction_path is not None:
        save_reconstruction(
            droid,
            traj_est,
            args.imagedir,
            args.calib,
            args.stride,
            args.t0,
            args.reconstruction_path,
        )
