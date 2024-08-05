RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

python ../bet/main_bet.py \
    --bet_arch bet_base \
    --encoder_arch resnet34 \
    --pretrained_weights IMAGENET1K_V1 \
    --data_path /ssd01/gagan/cpt_data/ours/jun19_bc_tabletop_rgb \
    --num_actions 8 \
    --naug 0 \
    --batch_size_per_gpu 32 \
    --epochs 128 \
    --output_dir $OUTDIR \
    --lr 0.0001 \
    --alpha 10 \
    --context_len 3