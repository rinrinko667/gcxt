import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils import frame_utils
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    # Force 3-channel RGB for robustness across PNG/JPG variants.
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, save_path=None):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    if save_path is not None:
        cv2.imwrite(save_path, img_flo[:, :, [2, 1, 0]])
        return

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey(1)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
        if args.flow_dir is not None:
            os.makedirs(args.flow_dir, exist_ok=True)

        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1_pad, image2_pad = padder.pad(image1, image2)

            _, flow_up = model(image1_pad, image2_pad, iters=20, test_mode=True)
            flow_up = padder.unpad(flow_up[0]).unsqueeze(0)

            stem1 = os.path.splitext(os.path.basename(imfile1))[0]
            stem2 = os.path.splitext(os.path.basename(imfile2))[0]
            outstem = f'{stem1}_{stem2}'

            if args.flow_dir is not None:
                flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
                frame_utils.writeFlow(os.path.join(args.flow_dir, f'{outstem}.flo'), flow_np)

            if args.save_dir is not None:
                viz(image1, flow_up, save_path=os.path.join(args.save_dir, f'{outstem}.png'))
            elif args.flow_dir is None:
                try:
                    viz(image1, flow_up)
                except cv2.error as err:
                    raise RuntimeError(
                        "OpenCV display is unavailable in current environment. "
                        "Please rerun with --save_dir to save visualization images."
                    ) from err


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--save_dir', help='save visualization images to a directory')
    parser.add_argument('--flow_dir', help='save raw optical flow .flo files to a directory')
    args = parser.parse_args()

    demo(args)
