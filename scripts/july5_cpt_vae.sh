RUNDIR="$PROJDIR/runs"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

for beta1 in 0.0001 0.0005; do
    for beta2 in 0.0001; do
            python $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
                --encoder_arch vit_tiny \
                --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_tabletop_mix_v2 \
                --output_dir $OUTDIR-beta1_$beta1-beta2_$beta2 \
                --beta1 $beta1 \
                --beta2 $beta2 \
                --alpha 0 \
                --core lapo \
                --epochs 200 \
                --skip_frames 20
    done
done
    

    