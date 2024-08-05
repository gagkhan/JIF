RUNDIR="$PROJDIR/runs/tabletop/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    exit 1
else
    OUTDIR=$RUNDIR/$1
fi

gpu=0

for batch_size in 32 64 128 ; do
    for out_dim in 4096 8192 16384; do
        for core in lapo ;  do
            for gc in True ; do
                export CUDA_VISIBLE_DEVICES=$gpu
                python $PROJDIR/CPT/cpt/main_cpt_dino.py \
                    --encoder_arch vit_tiny \
                    --data_path $PROJDIR/data/ours/ours_tabletop/ours_moveT_robot_frames \
                    --output_dir $OUTDIR-$core-batch-size-$batch_size-outdim-$out_dim \
                    --beta1 0.0001 \
                    --beta2 0.0001 \
                    --latent_action_dim 16 \
                    --latent_action_dim 3 \
                    --alpha 0 \
                    --core $core \
                    --epochs 200 \
                    --skip_frames 20 \
                    --goal_cond $gc \
                    --batch_size_per_gpu $batch_size \
                    --out_dim $out_dim &
                gpu=$((gpu + 1))
            done
        done
    done
done