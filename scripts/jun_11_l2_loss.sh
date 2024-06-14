RUNDIR="$PROJDIR/runs/tabletop"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi

export CUDA_VISIBLE_DEVICES=4
ladim=16
goal_cond=False
norm_last_layer=False

for outdim in 16 32 64 128 256; do

    python ../visual/main_cpt.py \
        --arch vit_tiny \
        --measure l2 \
        --data_path $PROJDIR/data/ours/ours_tabletop/ours_tabletop_mix_v2 \
        --local_crops_scale 0.99 1.0 \
        --global_crops_scale 0.99 1.0 \
        --epochs 100 \
        --output_dir $OUTDIR/"goal_cond_${goal_cond}_norm_last_${norm_last_layer}"/outdim_$outdim \
        --lr 0.001 \
        --latent_action_dim $ladim \
        --alpha 0 \
        --goal_cond $goal_cond \
        --norm_last_layer $norm_last_layer \
        --out_dim $outdim
    
done