RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

python ../../../bc/main_bc.py \
    --decoder_arch mlp_large \
    --encoder_arch resnet34 \
    --pretrained_weights IMAGENET1K_V1 \
    --data_path /ssd01/gagan/cpt_data/ours/jun19_bc_tabletop_rgb \
    \
    --naug 0 \
    --batch_size_per_gpu 128 \
    --epochs 50 \
    --output_dir $OUTDIR \
    --lr 0.001 \
    --alpha 10 \
    \
    --use_ee \