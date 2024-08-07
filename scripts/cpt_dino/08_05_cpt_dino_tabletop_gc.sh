RUNDIR="$PROJDIR/runs/tabletop/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    exit 1
else
    OUTDIR=$RUNDIR/$1
fi

gpu=4
for core in lapo;  do
    for gc in True False; do
        export CUDA_VISIBLE_DEVICES=$gpu
        python $PROJDIR/CPT/cpt/main_cpt_dino.py \
            --encoder_arch vit_tiny \
            --data_path $PROJDIR/data/ours/ours_tabletop/ours_moveT_robot_frames \
            --output_dir $OUTDIR/$core-gc-$gc \
            --beta1 0.0001 \
            --beta2 0.0001 \
            --latent_action_dim 3 \
            --latent_state_dim 16 \
            --alpha 0 \
            --core $core \
            --epochs 400 \
            --skip_frames 20 \
            --goal_cond $gc \
            --momentum_teacher 0.9996 &
        echo "waiting for 60s"
        sleep 60
        gpu=$((gpu + 1))
    done
done
