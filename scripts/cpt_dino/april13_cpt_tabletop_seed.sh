#!/bin/bash
RUNDIR="../../runs/ours/tabletop"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

for seed in 0 1 2; do
    python ../visual/main_cpt.py \
        --arch vit_small \
        --measure cross_entropy \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_tabletop_mix_v2 \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --local_crops_number 0 \
        --epochs 100 \
        --output_dir $OUTDIR/seed-$seed\
        --lr 0.001 \
        --latent_action_dim 16 \
        --seed $seed \
        --alpha 10 \
        --pretrained_weights /home/gagan/Home/VideoIL/dino_deitsmall16_pretrain_full_checkpoint.pth
done




