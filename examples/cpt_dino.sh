RUNDIR="$PROJDIR/runs"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

rm -rf $OUTDIR
python $PROJDIR/CPT/cpt/main_cpt_dino.py \
    --encoder_arch vit_tiny \
    --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_tabletop_mix_v2 \
    --output_dir $OUTDIR \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --alpha 0 \
    --core ilpo
    # --batch_size_per_gpu 64