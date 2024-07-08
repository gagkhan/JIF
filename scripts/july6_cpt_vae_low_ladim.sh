RUNDIR="$PROJDIR/runs/tabletop/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

gpu=0
for core in lapo;  do
    for lsdim in 16 12 8 4; do
        for ladim in 3; do
            for gc in True False; do
                python $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
                    --encoder_arch vit_tiny \
                    --data_path $PROJDIR/data/ours/ours_tabletop/ours_moveT_robot_frames \
                    --output_dir $OUTDIR-$core-gc-$gc-lsdim-$lsdim-ladim-$ladim\
                    --beta1 0.0001 \
                    --beta2 0.0001 \
                    --latent_action_dim $ladim \
                    --alpha 0 \
                    --core $core \
                    --epochs 200 \
                    --skip_frames 20 \
                    --gpu $gpu \
                    --goal_cond $gc &
                gpu=$((gpu + 1))
            done
        done
    done
done
        

    