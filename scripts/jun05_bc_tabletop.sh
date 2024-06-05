RUNDIR="../../runs/tabletop"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

python ../visual/main_bc.py \
    --arch vit_tiny \
    --patch_size 8 \
    --data_path /home/roam-bimm/roam_bimm_ws/src/data_collection/scripts/demonstrations/jun04_bc_tabletop \
    --local_crops_scale 0.99 1.0 \
    --global_crops_scale 0.99 1.0 \
    --local_crops_number 0 \
    --batch_size_per_gpu 24 \
    --epochs 50 \
    --output_dir $OUTDIR \
    --lr 0.001 \
    --alpha 10
