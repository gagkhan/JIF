RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

python ../../cpt/main_cpt_pure_bc.py \
    --data_path /ssd01/gagan/cpt_data/ours/10_15_pickrobot \
    --output_dir $OUTDIR \
    --use_cam2    True \
    --use_cam3    True \
    --use_tactile False \
    --use_ee      True \
    --goal_cond   True \
    --batch_size_per_gpu 32 \
    --beta 0.0 \
    --alpha 1.0 \
    --epochs 100 \
    --encoder_arch vitact_tiny \
    --train_split 0.7 \
    --num_workers 16 \