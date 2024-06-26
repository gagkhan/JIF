RUNDIR="$PROJDIR/runs"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi


python $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
    --encoder_arch resnet34 \
    --data_path /home/gagan/Home/VideoIL/data/ours/ours_tabletop/ours_tabletop_mix_v2 \
    --output_dir $OUTDIR


