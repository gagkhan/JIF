RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    # OUTDIR=/home/sarahp/VideoIL/runs/main_cpt_vitact_dino2/dynamo_best_lapo_epoch50
    OUTDIR=/home/sarahp/VideoIL/runs/robot_train_pretrained_weights/dynamo_baseline
    # OUTDIR=/home/sarahp/VideoIL/runs/09_17_v0.1_step1_experiments/l2_vitact_tiny_batch128_center
else
    OUTDIR=$RUNDIR/$1
fi

rm -rf $OUTDIR
python $PROJDIR/CPT/cpt/main_cpt_vitact_dino.py \
    --encoder_arch vitact_tiny \
    --data_path /ssd01/gagan/cpt_data/ours/10_15_pickrobot \
    --output_dir $OUTDIR \
    --simloss dynamo \
    --skip_frames 10 \
    --center_update False \
    --quantize_state False \
    --quantize_action False \
    --alpha 0 \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --momentum_teacher  0.9995 \
    --epochs 100 \
    --core lapo \
    --batch_size_per_gpu 128 \
    --lr 5.5e-5 \
    --use_cam2    True \
    --use_cam3    True \
    --use_tactile False \
    --use_ee      True \
    --loss_after_head False \
    --pretrained_weights  /home/sarahp/VideoIL/runs/clean_dataset/dynamo_best/checkpoint_best.pth\
