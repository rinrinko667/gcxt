

to run egoego with our own data，please turn to scripts/stage12

- Stage1 (HeadNet / GravityNet): data prep, train, eval, infer
- Stage2 (Diffusion): data prep, train, eval, infer


The author used [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) to extract camera poses. They also provided results of DROID-SLAM for ARES, Kinpoly-MoCap, GIMO [here](https://drive.google.com/drive/folders/1hoWdQKXoX4Hc7FGtNcJqmo6fO9QoGKEw?usp=drive_link). Please find the results for each dataset and put them into desired path ```data/ares/droid_slam_res/```, ```data/gimo/droid_slam_res/```, ```data/kinpoly-mocap/droid_slam_res```, ```data/kinpoly-realworld/droid_slam_res/```. 

 [RAFT](https://github.com/princeton-vl/RAFT) to extract optical flow, we provided optical flow features extracted using a pre-trained ResNet [here](https://drive.google.com/drive/folders/1hoWdQKXoX4Hc7FGtNcJqmo6fO9QoGKEw?usp=drive_link). Please find the results for each dataset and put them into desired path ```data/ares/raft_of_feats```, ```data/gimo/raft_of_feats```, ```data/kinpoly/fpv_of_feats```. 

For our own data, I first use the DROID-SLAM and RAFT to extract the features I need


Custom stage1 default path now assumes:

- flow: `flow_out/*.flo`
- slam: `output_dir/{seq_name}/reconstruction.pth` 






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


