#!/bin/bash

python ../visual/main_dino.py \
    --arch vit_tiny \
    --measure cross_entropy \
    --data_path /home/gagan/Home/VideoIL/data/ours/ours_v3_frames \
    --epochs 2000 \
    --output_dir ../../DINO/test