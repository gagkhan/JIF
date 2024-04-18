#!/bin/bash

python ../visual/offline_eval.py \
    --checkpoint_key student \
    --arch vit_small \
    --pretrained_weights /home/gagan/Home/VideoIL/runs/tabletop/april17_bc/checkpoint.pth \
    --data_root /home/gagan/Home/VideoIL/data/eval/tabletop_robot/moveT \
    --algo bc