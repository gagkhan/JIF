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
    --data_path $PROJDIR/data/bridge/raw/bridge_data_v1 \
    --output_dir $OUTDIR/lapo_gc_True \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --alpha 0 \
    --core lapo \
    --batch_size_per_gpu 64 \
    --train_split 0.95 \
    --num_workers 10 \
    --warmup_epochs 0 \
    --goal_cond True &

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_TRAINERS \
    $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
    --encoder_arch vit_small \
    --data_path $PROJDIR/data/bridge/raw/bridge_data_v1 \
    --output_dir $OUTDIR/ilpo_gc_True \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --alpha 0 \
    --core ilpo \
    --batch_size_per_gpu 64 \
    --train_split 0.95 \
    --num_workers 10 \
    --warmup_epochs 0 \

