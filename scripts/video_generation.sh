#!/bin/bash

# loop over videos in the dataset
MODEL_PATH=/home/gagan/Home/VideoIL/runs/tabletop/april22_act_dec_clip_grad_vit_tiny_crops/ladim-16
VIDEO_DIR=/home/gagan/Home/VideoIL/data/evalvideo/tabletop_robot/moveT

OUTPUT_DIR=$MODEL_PATH/attn/videos

mkdir -p $OUTPUT_DIR

for video in $VIDEO_DIR/*; do
    echo "Processing $video"
    output_path="$OUTPUT_DIR/$(basename $video .mp4)"
    echo "Output path: $output_path"
    # random weights
    # python ../visual/video_generation.py --arch vit_tiny --input_path $video --patch_size 16 --output_path $output_path 
    
    # model weights
    python ../visual/video_generation.py --arch vit_tiny --input_path $video --patch_size 16 --output_path $output_path --pretrained_weights $MODEL_PATH/checkpoint.pth
    
    # pretrained weights
    # python ../visual/video_generation.py --arch vit_small --input_path $video --patch_size 16 --output_path $output_path --pretrained_weights /home/gagan/Home/VideoIL/dino_pretrained/dino_deitsmall16_pretrain_full_checkpoint.pth
done
