#!/bin/bash

python ../visual/main_cpt.py \
    --arch vit_tiny \
    --measure cross_entropy \
    --data_path /home/gagan/Home/VideoIL/data/ours/ours_v2_frames \
    --local_crops_scale 0.4 1.0 \
    --epochs 2000 \
    --output_dir ../../DINO/test2
    