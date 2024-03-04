#!/bin/bash

python ../visual/main_dino.py \
    --arch vit_tiny \
    --measure cross_entropy \
    --data_path /home/gagan/Home/VideoIL/data/ssv2/20bn-something-something-v2-frames-tiny \
    --epochs 2000 \
    --output_dir ../../DINO/test