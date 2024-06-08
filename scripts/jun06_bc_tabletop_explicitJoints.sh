RUNDIR="../../runs/tabletop"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

python ../visual/main_bc_explicitJoints.py \
    --arch vit_tiny \
    --patch_size 16 \
    --data_path /home/roam-bimm/roam_bimm_ws/src/data_collection/scripts/demonstrations/jun06_bc_tabletop_rgb \
    --local_crops_scale 0.99 1.0 \
    --global_crops_scale 0.99 1.0 \
    --local_crops_number 0 \
    --batch_size_per_gpu 32 \
    --epochs 100 \
    --output_dir $OUTDIR \
    --lr 0.001 \
    --alpha 10
