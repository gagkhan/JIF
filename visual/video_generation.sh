#!/bin/bash

# loop over videos in the dataset
MODEL_PATH=/home/gagan/Home/VideoIL/DINO/feb29_ours_v3_cpt

VIDEO_DIR=/home/gagan/Home/VideoIL/data/ours_v3

OUTPUT_DIR=$MODEL_PATH/attn/videos

mkdir -p $OUTPUT_DIR

for video in $VIDEO_DIR/*; do
    echo "Processing $video"
    output_path="$OUTPUT_DIR/$(basename $video .mp4)"
    echo "Output path: $output_path"
    python video_generation.py --arch vit_tiny --input_path $video --patch_size 16 --output_path $output_path --pretrained_weights $MODEL_PATH/checkpoint.pth
done
