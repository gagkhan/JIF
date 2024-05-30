#!/bin/bash
RUNDIR="../../runs/tabletop"


if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    OUTDIR=$RUNDIR/debug
else
    OUTDIR=$RUNDIR/$1
fi


for codebook_len in 16 32 48 64; do

    for embed_dim in 4 8 12; do
    python ../visual/main_vqbet_vq.py \
            --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_moveT_robot_frames \
            --epochs 10 \
            --output_dir $OUTDIR \
            --codebook_len $codebook_len \
            --embed_dim $embed_dim
    done
done
