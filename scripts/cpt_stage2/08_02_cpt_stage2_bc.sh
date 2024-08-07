RUNDIR="$PROJDIR/runs/tabletop"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

for alpha in 1 10 50; do
    python $PROJDIR/CPT/cpt/main_cpt_stage2_bc.py \
        --teacher_chkpt $PROJDIR/runs/tabletop/07_25_cpt_val_loss_tabletop-lapo-gc-True-lsdim-16-ladim-3/checkpoint.pth \
        --data_path $PROJDIR/data/ours/ours_tabletop/ours_moveT_robot_frames \
        --output_dir $OUTDIR \
        --beta 1 \
        --alpha $alpha
done