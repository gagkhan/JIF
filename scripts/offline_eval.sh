#!/bin/bash

python ../visual/offline_eval.py \
    --checkpoint_key student \
    --arch vit_tiny \
    --pretrained_weights /home/gagan/Home/VideoIL/runs/tabletop/april22_act_dec_clip_grad_vit_tiny_crops/ladim-16/checkpoint.pth \
    --data_root /home/gagan/Home/VideoIL/data/eval/tabletop_robot/moveT \
    --algo cpt

# python ../visual/offline_eval.py \
#     --checkpoint_key student \
#     --arch vit_small \
#     --pretrained_weights /home/gagan/Home/VideoIL/runs/tabletop/april17_bc/checkpoint.pth \
#     --data_root /home/gagan/Home/VideoIL/data/eval/tabletop_robot/moveT \
#     --algo bc
# 
