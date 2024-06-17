RUNDIR="/proj/vondrick4/gagan/VideoIL/runs/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi


for ladim in 32 16; do
    for i in 0 1 2; do
        python ../visual/main_cpt.py \
            --arch vit_tiny \
            --measure cross_entropy \
            --data_path /proj/vondrick4/gagan/VideoIL/data/ours/ours_tabletop/ours_tabletop_mix_v2 \
            --local_crops_scale 0.99 1.0 \
            --global_crops_scale 0.99 1.0 0\
            --epochs 100 \
            --output_dir $OUTDIR/ladim-$ladim/trial-$i \
            --lr 0.001 \
            --latent_action_dim $ladim \
            --alpha 0 \
            --latent_action_cond False \
            --beta 0
    done
done