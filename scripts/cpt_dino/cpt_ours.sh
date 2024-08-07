#!/bin/bash
RUNDIR="../../runs/ours"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

for ladim in 2 32; do
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
        --alpha 0.1
done


for skip_frames in 4 3 2 1 0; do
    python ../visual/main_cpt.py \
        --arch vit_tiny \
        --measure cross_entropy \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_v2_frames \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --epochs 100 \
        --output_dir $OUTDIR/ours_v2_noact_skip_frames-$skip_frames \
        --lr 0.001 \
        --skip_frames $skip_frames \
        --alpha 0
done

for skip_frames in 4 3 2 1 0; do
    python ../visual/main_cpt.py \
        --arch vit_tiny \
        --measure cross_entropy \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_v2_frames \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --epochs 100 \
        --output_dir $OUTDIR/ours_v2_skip_frames-$skip_frames \
        --lr 0.001 \
        --skip_frames $skip_frames \
        --alpha 0.1
done

for skip_frames in 4 3 2 1 0; do
    python ../visual/main_cpt.py \
        --arch vit_tiny \
        --measure cross_entropy \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_v3_frames \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --epochs 100 \
        --output_dir $OUTDIR/ours_v3_skip_frames-$skip_frames \
        --lr 0.001 \
        --skip_frames $skip_frames \
        --alpha 0.1
done





