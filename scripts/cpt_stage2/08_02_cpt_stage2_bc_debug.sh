RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

python ../../cpt/main_cpt_stage2_bc.py \
    --teacher_chkpt /ssd01/gagan/cpt_checkpoints/08_19_cpt_dino_pick_v0.3/checkpoint.pth \
    --data_path /ssd01/gagan/cpt_data/ours/aug09_pickhuman \
    --output_dir $OUTDIR \
    --use_cam2    True \
    --use_cam3    True \
    --use_tactile True \
    --use_ee      False \
    --beta 1 \
    --alpha 1