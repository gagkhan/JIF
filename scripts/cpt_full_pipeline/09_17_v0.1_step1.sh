RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    # OUTDIR=/home/sarahp/VideoIL/runs/09_17_v0.1_step1_experiments/ce_quantize #in cross entropy tmux tab and l2 loss with softmax on student
    OUTDIR=/home/sarahp/VideoIL/runs/09_17_v0.1_step1_experiments/l2_no_softmax_quantize_action
else
    OUTDIR=$RUNDIR/$1
fi

rm -rf $OUTDIR
python $PROJDIR/CPT/cpt/main_cpt_vitact_dino.py \
    --encoder_arch vitact_small \
    --data_path /ssd01/gagan/cpt_data/ours/aug09_pickhuman \
    --output_dir $OUTDIR \
    --simloss l2 \
    --skip_frames 10 \
    --quantize_state True \
    --quantize_action False \
    --alpha 0 \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --momentum_teacher 0.9995 \
    --epochs 100 \
    --core lapo \
    --batch_size_per_gpu 64 \
    --use_cam2    True \
    --use_cam3    True \
    --use_tactile False \
    --use_ee      False \
