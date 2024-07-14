RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

python ../bet/vq_actions.py \
    --data_path /ssd01/gagan/cpt_data/ours/jun19_bc_tabletop_rgb \
    --batch_size_per_gpu 16 \
    --epochs 512 \
    --output_dir $OUTDIR \
    --lr 0.0001 \
    --min_lr 0.0000001 \