#!/bin/bash
RUNDIR="../../runs/bimanual/v2/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

for ladim in 32 16; do
    python ../visual/main_cpt.py \
        --arch vit_tiny \
        --measure cross_entropy \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_v2_frames \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --epochs 100 \
        --output_dir $OUTDIR/ladim-$ladim \
        --lr 0.001 \
        --latent_action_dim $ladim \
        --alpha 0
done


for ladim in 32 16; do
    python ../visual/main_cpt.py \
        --arch vit_small \
        --measure cross_entropy \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_v2_frames \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --local_crops_number 0 \
        --epochs 100 \
        --output_dir $OUTDIR/ladim-$ladim \
        --lr 0.001 \
        --latent_action_dim $ladim \
        --alpha 0 \
        --pretrained_weights /home/gagan/Home/VideoIL/dino_deitsmall16_pretrain_full_checkpoint.pth
done





