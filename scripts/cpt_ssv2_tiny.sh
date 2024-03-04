#!/bin/bash

python ../visual/main_cpt.py \
    --arch vit_tiny \
    --measure cross_entropy \
    --data_path /home/gagan/Home/VideoIL/data/ssv2/20bn-something-something-v2-frames-tiny \
    --local_crops_number 0.4 1.0 \
    --epochs 2000 \
    --output_dir ../../DINO/test