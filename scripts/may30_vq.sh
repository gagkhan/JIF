#!/bin/bash
RUNDIR="../../runs/tabletop"


if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    OUTDIR=$RUNDIR/debug
else
    OUTDIR=$RUNDIR/$1
fi


for codebook_len in 32; do
    for embed_dim in 4; do
    python ../visual/main_vqbet_vq.py \
            --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_moveT_robot_frames \
            --epochs 100 \
            --output_dir $OUTDIR/embed_dim-$embed_dim/codebook_len-$codebook_len \
            --codebook_len $codebook_len \
            --embed_dim $embed_dim \
            --lr 1e-4
    done
done
