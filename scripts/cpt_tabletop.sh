#!/bin/bash
RUNDIR="../../runs/ours"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

for ladim in 32 16 64 8 ; do
    python ../visual/main_cpt.py \
        --arch vit_tiny \
        --measure cross_entropy \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop_mix_v2 \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --epochs 100 \
        --output_dir $OUTDIR/ladim-$ladim \
        --lr 0.001 \
        --latent_action_dim $ladim \
        --alpha 10
done

for ladim in 32 16 64 8 ; do
    python ../visual/main_cpt.py \
        --arch vit_tiny \
        --measure cross_entropy \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop_mix_v2 \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --epochs 100 \
        --output_dir $OUTDIR/beta=0_ladim-$ladim \
        --lr 0.001 \
        --latent_action_dim $ladim \
        --alpha 10 \
        --beta 0
done




