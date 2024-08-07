RUNDIR="$PROJDIR/runs"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

rm -rf $OUTDIR

NUM_TRAINERS=8
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_TRAINERS \
    $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
    --encoder_arch vit_small \
    --data_path $PROJDIR/data/bridge/raw/ \
    --output_dir $OUTDIR \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --alpha 0 \
    --core lapo \
    