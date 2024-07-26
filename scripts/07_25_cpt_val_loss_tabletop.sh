RUNDIR="$PROJDIR/runs/tabletop/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    exit 1
else
    OUTDIR=$RUNDIR/$1
fi

gpu=2

for lsdim in 16 8; do
    for ladim in 3; do
        for core in lapo ilpo;  do
            for gc in True False; do
                export CUDA_VISIBLE_DEVICES=$gpu
                python $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
                    --encoder_arch vit_tiny \
                    --data_path $PROJDIR/data/ours/ours_tabletop/ours_moveT_robot_frames \
                    --output_dir $OUTDIR-$core-gc-$gc-lsdim-$lsdim-ladim-$ladim\
                    --beta1 0.0001 \
                    --beta2 0.0001 \
                    --latent_action_dim $ladim \
                    --alpha 0 \
                    --core $core \
                    --epochs 100 \
                    --skip_frames 5 \
                    --goal_cond $gc
                # gpu=$((gpu + 1))
            done
        done
    done
done