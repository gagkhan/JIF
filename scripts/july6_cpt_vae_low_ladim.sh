RUNDIR="$PROJDIR/runs/tabletop/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

for core in lapo;  do
    for lsdim in 16 12 8 4; do
        for ladim in 2 3; do
            python $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
                --encoder_arch vit_tiny \
                --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_moveT_robot_frames \
                --output_dir $OUTDIR-$core-ladim-$ladim\
                --beta1 0.0001 \
                --beta2 0.0001 \
                --latent_action_dim $ladim \
                --alpha 0 \
                --core $core \
                --epochs 200 \
                --skip_frames 20
        done
done
        

    