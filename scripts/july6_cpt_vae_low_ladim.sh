RUNDIR="$PROJDIR/runs/tabletop/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

# IMPORTANT: Have you undone changes liel x_goal *= 0 to enable goal conditioning????

for core in lapo ilpo;  do
    for ladim in 2 3 4; do
        python $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
            --encoder_arch vit_tiny \
            --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_tabletop_mix_v2 \
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
        

    