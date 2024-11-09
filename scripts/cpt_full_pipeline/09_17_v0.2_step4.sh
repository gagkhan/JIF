RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    OUTDIR=$RUNDIR/debug
else
    OUTDIR=$RUNDIR/$1
fi

rm -rf $OUTDIR
python ../../cpt/main_cpt_stage2_bc.py \
    --teacher_chkpt /ssd01/gagan/cpt_checkpoints/09_17_v0.2_step3/checkpoint.pth \
    --pretrained_weights /ssd01/gagan/cpt_checkpoints/09_17_v0.2_step2/checkpoint.pth \
    --data_path /ssd01/gagan/cpt_data/ours/aug09_pickhuman \
    --output_dir $OUTDIR \
    --alpha 1.0 \
    --beta 1.0 \
    --use_cam2    True \
    --use_cam3    True \
    --use_tactile False \
    --use_ee      True \
    --batch_size_per_gpu 128 \