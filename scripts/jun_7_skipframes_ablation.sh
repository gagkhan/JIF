RUNDIR="/proj/vondrick4/gagan/VideoIL/runs"
# RUNDIR="../../runs"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
else
    OUTDIR=$RUNDIR/$1
fi


# export CUDA_VISIBLE_DEVICES=2
# latent_action_cond=False
# i=0
# ladim=16
# skip_frames=20

# export CUDA_VISIBLE_DEVICES=3
# latent_action_cond=False
# i=1
# ladim=16
# skip_frames=20

# export CUDA_VISIBLE_DEVICES=4
# latent_action_cond=False
# i=2
# ladim=16
# skip_frames=20

# export CUDA_VISIBLE_DEVICES=5
# latent_action_cond=False
# i=0
# ladim=16
# skip_frames=40

# export CUDA_VISIBLE_DEVICES=6
# latent_action_cond=False
# i=1
# ladim=16
# skip_frames=40

export CUDA_VISIBLE_DEVICES=7
latent_action_cond=False
i=2
ladim=16
skip_frames=40

python ../visual/main_cpt.py \
    --arch vit_tiny \
    --measure cross_entropy \
    --data_path /proj/vondrick4/gagan/VideoIL/data/ours/ours_tabletop/ours_tabletop_mix_v2 \
    --local_crops_scale 0.99 1.0 \
    --global_crops_scale 0.99 1.0 \
    --epochs 100 \
    --output_dir $OUTDIR/skipframes_$skip_frames/trial-$i \
    --lr 0.001 \
    --latent_action_dim $ladim \
    --alpha 0 \
    --latent_action_cond $latent_action_cond \
    --goal_cond False \
    --skip_frames $skip_frames


