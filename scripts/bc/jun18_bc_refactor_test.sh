RUNDIR="../../runs/"

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
    --data_path /ssd01/gagan/cpt_data/ours/jun09_bc_tabletop_rgb_simple \
    --naug 2 \
    --use_ee \
    --batch_size_per_gpu 32 \
    --epochs 100 \
    --output_dir $OUTDIR \
    --lr 0.001 \
    --alpha 10