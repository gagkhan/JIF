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
python $PROJDIR/CPT/cpt/main_cpt_stage2_bc.py \
    --teacher_chkpt $RUNDIR/tabletop/07_25_cpt_val_loss_tabletop-lapo-gc-True-lsdim-16-ladim-3/checkpoint.pth \
    --data_path $PROJDIR/data/ours/ours_tabletop/ours_moveT_robot_frames \
    --output_dir $OUTDIR \
    --beta 0.0001 \
    --alpha 0 \
