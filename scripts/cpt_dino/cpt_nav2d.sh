#!/bin/bash
RUNDIR="../../runs/nav2d"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi
  
    python ../visual/main_cpt.py \
    --arch vit_tiny \
    --measure l2 \
    --data_path /home/gagan/Home/VideoIL/data/nav2d/visual_v1 \
    --output_dir $OUTDIR \
    --alpha 0 \
    --global_crops_scale 0.99 1.0 \
    --local_crops_scale 0.99 1.0 \
    --epochs 2000
   
    # python ../visual/main_cpt.py \
    # --arch vit_tiny \
    # --measure l2 \
    # --data_path /home/gagan/Home/VideoIL/data/nav2d/visual_v1 \
    # --output_dir $OUTDIR \
    # --alpha 10 \
    # --global_crops_scale 0.99 1.0 \
    # --local_crops_number 0 \
    # --norm_last_layer False \
    # --out_dim 2 \
    # --latent_action_dim 2 \
    # --warmup_epochs 1 \
    # # --local_crops_scale 0.99 1.0 \
    # --latent_action_dim 128 \
    # --beta 0.001 \
    # --skip_frames 1 \
    # --epochs 2000\
    # --lr 0.01 
    #  --local_crops_number 0 \
fi
