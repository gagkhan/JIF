RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    OUTDIR=$RUNDIR/debug
else
    OUTDIR=$RUNDIR/$1
fi

rm -rf $OUTDIR
python ../../../cpt/main_cpt_vitact_dino.py \
    --encoder_arch vitact_small \
    --data_path /ssd01/gagan/cpt_data/ours/aug09_pickhuman \
    --output_dir $OUTDIR \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --alpha 0 \
    --core lapo \
    --batch_size_per_gpu 64