#!/bin/bash
RUNDIR="../../runs/ours"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi


# python ../visual/main_cpt.py \
#     --arch vit_tiny \
#     --measure cross_entropy \
#     --data_path /home/gagan/Home/VideoIL/data/ours/ours_v2_frames \
#     --local_crops_scale 0.99 1.0 \
#     --local_crops_scale 0.99 1.0 \
#     --epochs 2000 \
#     --output_dir $OUTDIR \

python ../visual/main_cpt.py \
    --arch vit_tiny \
    --measure cross_entropy \
    --data_path /home/gagan/Home/VideoIL/data/ours/ours_v2_frames \
    --local_crops_scale 0.99 1.0 \
    --local_crops_scale 0.99 1.0 \
    --epochs 100 \
    --output_dir $OUTDIR \
    --lr 0.00075 \