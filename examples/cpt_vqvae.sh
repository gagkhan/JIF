RUNDIR="$PROJDIR/runs"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided, using debug"
    OUTDIR=$RUNDIR/debug
else
    OUTDIR=$RUNDIR/$1
fi


# --data_path $PROJDIR/data/ssv2/20bn-something-something-v2-frames-tiny \

rm -rf $OUTDIR
python $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
    --encoder_arch vit_tiny \
    --data_path $PROJDIR/data/ours/ours_tabletop/ours_moveT_robot_frames \
    --output_dir $OUTDIR \
    --beta1 0.0001 \
    --beta2 0.0001 \
    --alpha 0 \
    --core lapo
