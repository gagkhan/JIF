RUNDIR="../../runs/tabletop"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

python ../visual/main_bc.py \
    --arch vit_tiny \
    --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_moveT_robot_frames \
    --local_crops_scale 0.99 1.0 \
    --global_crops_scale 0.99 1.0 \
    --local_crops_number 0 \
    --epochs 100 \
    --output_dir $OUTDIR \
    --lr 0.001 \
    --alpha 10