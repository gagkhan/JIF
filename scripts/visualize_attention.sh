#!/bin/bash

# Visualize attention maps

# MODEL_PATH=/home/gagan/Home/VideoIL/DINO/feb14_cpt_nolocal
# MODEL_PATH=/home/sarahp/VideoIL/runs/full_dataset/dynamo_best_tactile_cam1_cam2
MODEL_PATH=/home/sarahp/VideoIL/runs/robot_train/big_dataset_pretrain_dynamo_best

SSV2_PATH=/home/gagan/Home/VideoIL/data/20bn-something-something-v2-frames-tiny
IMAGE2="robot_data/demo_1"
IMAGE3="2272/5"
IMAGE=$IMAGE3

# MODEL_PATH=/home/gagan/Home/VideoIL/DINO/feb23_2000
# OURS_PATH=/home/gagan/Home/VideoIL/data/ours_v2_frames
# IMAGE1="twohand__plu/000001"
# IMAGE2="twohand_book/000001"
# IMAGE3="twohand_transfersoccer/000001"
# IMAGE4="twohand_stacklego1/000001"
# IMAGE5="twohand_insertlego/000001"
# IMAGE6="twohands_batter/000001"
# IMAGE7="twohand_openboo/000013"
# IMAGE=$IMAGE6
# IMAGE_PATH=$OURS_PATH/$IMAGE.jpg

OUTPUT_DIR=$MODEL_PATH/attn/$IMAGE
# python /home/sarahp/VideoIL/CPT/visuotactile/visualize_attention.py --encoder_arch vitact_small  --data_path $IMAGE_PATH --patch_size 16 --output_dir $OUTPUT_DIR --pretrained_weights $MODEL_PATH/checkpoint.pth
python /home/sarahp/VideoIL/CPT/visuotactile/visualize_attention.py --encoder_arch vitact_tiny --patch_size 16 --output_dir $OUTPUT_DIR --pretrained_weights $MODEL_PATH/checkpoint_best.pth --use_tactile False --demo_num 5
# python /home/sarahp/VideoIL/CPT/visuotactile/visualize_attention.py --encoder_arch vitact_tiny --patch_size 16 --output_dir $OUTPUT_DIR --pretrained_weights $MODEL_PATH/checkpoint.pth --use_tactile False
# python /home/sarahp/VideoIL/CPT/visual/visualize_attention.py --encoder_arch vitact_small --patch_size 16 --output_dir $OUTPUT_DIR --pretrained_weights $MODEL_PATH/checkpoint.pth --use_tactile False
