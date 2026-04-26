# Stage1 / Stage2 Scripts

This folder contains all scripts for:

- Stage1 (HeadNet / GravityNet): data prep, train, eval, infer
- Stage2 (Diffusion): data prep, train, eval, infer


# 1) 数据转换（stage1 + stage2）
sh scripts/stage12/prepare_mydataset_stage1.sh
sh scripts/stage12/prepare_mydataset_stage2.sh

# 2) Stage1 训练
sh scripts/stage12/train_headnet_on_custom.sh
sh scripts/stage12/train_gravitynet_on_custom.sh

# 3) Stage1 评估
sh scripts/stage12/eval_headnet_on_custom.sh
sh scripts/stage12/eval_gravitynet_on_custom.sh

# 4) Stage2 训练与评估
sh scripts/stage12/train_stage2_on_custom.sh
sh scripts/stage12/eval_stage2_on_custom.sh

# 5) 推理（分步）
sh scripts/stage12/infer_headnet_on_custom.sh
sh scripts/stage12/infer_gravitynet_on_custom.sh
sh scripts/stage12/infer_stage2_from_stage1_custom.sh

# 6) 一键整体评估（HeadNet+GravityNet+Stage2）
sh scripts/stage12/eval_egogeo_custom.sh

Custom stage1 default path now assumes:

- flow: `flow_out/*.flo`
- slam: `output_dir/{seq_name}/reconstruction.pth` (also supports `{seq_name}.pth`, `{seq_name}.npy`, and fallback `output_dir/reconstruction.pth`)


