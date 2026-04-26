# Custom Data Pipeline

This folder centralizes all code for running EgoEgo with your own dataset.

## Data Conversion

- `convert_mydataset_to_egoego_stage1.py`
- `convert_mydataset_to_egoego_stage2.py`
- `custom_headpose_dataset.py`

`convert_mydataset_to_egoego_stage1.py` now supports official-style stage1 inputs:

- camera GT head pose from `mydataset/<seq>/cam/*.npz`
- optical flow from `flow_out/*.flo` (or `.npy`)
- SLAM trajectory from `output_dir/reconstruction.pth` (or `.npy`)

By default, it writes raw flow maps (`of`) resized to `224x224`, so HeadNet scripts should run **without**
`--input_of_feats`.  
If you already have pre-extracted flow features (`raft_of_feats`-style 1D vectors), run conversion with
`--use_of_feats` and then enable `--input_of_feats` in train/eval/infer scripts.

## Stage1: HeadNet / GravityNet

- Train:
  - `train_headnet_custom.py`
  - `train_gravitynet_custom.py`
- Eval:
  - `eval_headnet_custom.py`
  - `eval_gravitynet_custom.py`
- Infer:
  - `infer_headnet_custom.py`
  - `infer_gravitynet_custom.py`

## Stage2: Diffusion Inference

- `infer_stage2_custom.py`

## Script Entrypoints

Use scripts in `scripts/stage12/`, for example:

- `sh scripts/stage12/prepare_mydataset_stage1.sh`
- `sh scripts/stage12/train_headnet_on_custom.sh`
- `sh scripts/stage12/train_gravitynet_on_custom.sh`
- `sh scripts/stage12/train_stage2_on_custom.sh`
