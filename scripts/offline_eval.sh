#!/bin/bash

python ../visual/offline_eval.py \
    --checkpoint_key student \
    --arch vit_small \
    --pretrained_weights /home/gagan/Home/VideoIL/runs/tabletop/apr13_finetune/ladim-64/checkpoint.pth \
    --data_root /home/gagan/Home/VideoIL/data/eval/tabletop_robot/moveT