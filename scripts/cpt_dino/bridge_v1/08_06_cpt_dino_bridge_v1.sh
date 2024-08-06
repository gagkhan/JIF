RUNDIR="$PROJDIR/runs/tabletop/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    exit 1
else
    OUTDIR=$RUNDIR/$1
fi

core=lapo
gc=True

NUM_TRAINERS=8
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$NUM_TRAINERS \
    $PROJDIR/CPT/cpt/main_cpt_dino.py \
    --encoder_arch vit_small \
    --data_path $PROJDIR/data/bridge/raw/bridge_data_v1  \
    --output_dir $OUTDIR/$core-gc-$gc \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --latent_action_dim 3 \
    --latent_state_dim 16 \
    --alpha 0 \
    --core $core \
    --epochs 100 \
    --skip_frames 5 \
    --goal_cond $gc \
    --batch_size_per_gpu 256
