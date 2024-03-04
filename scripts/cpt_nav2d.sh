#!/bin/bash

python ../visual/main_cpt.py \
    --arch vit_tiny \
    --measure l2 \
    --data_path /home/gagan/Home/VideoIL/data/nav2d_visual \
    --local_crops_number 0 \
    --global_crops_scale 0.95 1.0 \
    --norm_last_layer False \
    --latent_action_dim 2 \
    --output_dir ../../DINO/test \
    --beta 0.0 \
