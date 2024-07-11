RUNDIR="$PROJDIR/runs"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi


# --data_path $PROJDIR/data/ours/ours_tabletop/ours_tabletop_mix_v2 \
rm -rf $OUTDIR

NUM_TRAINERS=8

# CUDA_VISIBLE_DEVICES="0,3"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_TRAINERS \
    $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
    --encoder_arch vit_tiny \
    --data_path $PROJDIR/data/ssv2/20bn-something-something-v2-frames-tiny \
    --output_dir $OUTDIR \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --alpha 0 \
    --core lapo
