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
    --alpha 0 \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --momentum_teacher 0.9995 \
    --epochs 200 \
    --core lapo \
    --batch_size_per_gpu 64