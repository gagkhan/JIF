#!/bin/bash
RUNDIR="../../runs/tabletop"


if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    OUTDIR=$RUNDIR/debug
else
    OUTDIR=$RUNDIR/$1
fi


python ../visual/main_vqbet_vq.py \
        --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_moveT_robot_frames \
        --epochs 100 \
        --output_dir $OUTDIR
