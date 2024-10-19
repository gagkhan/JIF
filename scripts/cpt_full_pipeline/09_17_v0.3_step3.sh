RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    OUTDIR=$RUNDIR/debug
else
    OUTDIR=$RUNDIR/$1
fi

rm -rf $OUTDIR
python ../../cpt/main_cpt_vitact_dino.py \
    --encoder_arch vitact_tiny \
    --pretrained_weights /ssd01/gagan/cpt_checkpoints/09_17_v0.3_step1/checkpoint_best.pth \
    --data_path /ssd01/gagan/cpt_data/ours/aug30_pickrobot \
    --output_dir $OUTDIR \
    --alpha 1.0 \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --momentum_teacher 0.9995 \
    --epochs 1 \
    --warmup_epochs 0 \
    --core lapo \
    --batch_size_per_gpu 64 \
    --use_cam2    True \
    --use_cam3    True \
    --use_tactile True \
    --use_ee      True \