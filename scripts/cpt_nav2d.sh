#!/bin/bash
RUNDIR="../../runs/nav2d"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1

    python ../visual/main_cpt.py \
    --arch vit_tiny \
    --measure l2 \
    --data_path /home/gagan/Home/VideoIL/data/nav2d/visual_v1 \
    --local_crops_number 0 \
    --global_crops_scale 0.95 1.0 \
    --norm_last_layer False \
    --latent_action_dim 2 \
    --output_dir $OUTDIR \
    --beta 0.1 \
    --skip_frames 1

fi