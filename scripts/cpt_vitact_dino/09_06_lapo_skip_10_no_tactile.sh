RUNDIR="$PROJDIR/runs/cpt_vitact_dino"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    OUTDIR=$RUNDIR/debug
else
    OUTDIR=$RUNDIR/$1
fi

rm -rf $OUTDIR
python $PROJDIR/CPT/cpt/main_cpt_vitact_dino.py \
    --encoder_arch vitact_tiny \
    --data_path $PROJDIR/data/ours/aug09_pickhuman \
    --output_dir $OUTDIR \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --alpha 0 \
    --core lapo \
    --batch_size_per_gpu 64 \
    --quantize_action False \
    --skip_frames 10 \
    --use_tactile False