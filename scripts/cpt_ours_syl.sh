#!/bin/bash
RUNDIR="../../runs/ours"

export CUDA_VISIBLE_DEVICES=0

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

# python ../visual/main_cpt.py \
#     --arch vit_small \
#     --measure cross_entropy \
#     --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop_mix_v1 \
#     --local_crops_scale 0.99 1.0 \
#     --global_crops_scale 0.99 1.0 \
#     --epochs 100 \
#     --output_dir $OUTDIR-pretrained \
#     --lr 0.001 \
#     --local_crops_number 0 \
#     --alpha 0.1 \
#     --batch_size 16 \
#     --pretrained_weights ../../dino_deitsmall16_pretrain_full_checkpoint.pth

# python ../visual/main_cpt.py \
#     --arch vit_small \
#     --measure cross_entropy \
#     --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop_mix_v1 \
#     --local_crops_scale 0.99 1.0 \
#     --global_crops_scale 0.99 1.0 \
#     --epochs 100 \
#     --output_dir $OUTDIR-scratch \
#     --lr 0.001 \
#     --local_crops_number 0 \
#     --alpha 0.1 \
#     --batch_size 16 \

for ladim in 16 8 32; do
    python ../visual/main_cpt.py \
        --arch vit_tiny \
        --measure cross_entropy \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop_mix_v2 \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --epochs 100 \
        --output_dir $OUTDIR-ladim-$ladim \
        --lr 0.001 \
        --local_crops_number 0 \
        --alpha 10 \
        --batch_size 64 \
        --latent_action_dim $ladim
done
