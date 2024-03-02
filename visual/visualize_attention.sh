#!/bin/bash

# Visualize attention maps

# MODEL_PATH=/home/gagan/Home/VideoIL/DINO/feb14_cpt_nolocal
MODEL_PATH=/home/gagan/Home/VideoIL/DINO/feb3_ssv2_tiny

SSV2_PATH=/home/gagan/Home/VideoIL/data/20bn-something-something-v2-frames-tiny
IMAGE1="16935/000011"
IMAGE2="1273/000001"
IMAGE3="2272/000011"
IMAGE=$IMAGE3
IMAGE_PATH=$SSV2_PATH/$IMAGE.jpg

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
python visualize_attention.py --arch vit_tiny  --image_path $IMAGE_PATH --patch_size 16 --output_dir $OUTPUT_DIR --pretrained_weights $MODEL_PATH/checkpoint.pth



