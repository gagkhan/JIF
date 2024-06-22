RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

python ../visual/main_bc.py \
    --arch resnet34 \
    --pretrained_weights IMAGENET1K_V1 \
    --patch_size 16 \
    --data_path /ssd01/gagan/cpt_data/ours/jun19_bc_tabletop_rgb \
    --use_ee \
    --local_crops_scale 0.99 1.0 \
    --global_crops_scale 0.99 1.0 \
    --local_crops_number 0 \
    --batch_size_per_gpu 32 \
    --epochs 10 \
    --output_dir $OUTDIR \
    --lr 0.0001 \
    --alpha 10