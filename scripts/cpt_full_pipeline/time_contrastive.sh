RUNDIR="/ssd01/gagan/cpt_checkpoints"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    OUTDIR=/home/sarahp/VideoIL/runs/time_contrastive/shuffle_epochs100_batch12d8
    # OUTDIR=/home/sarahp/VideoIL/runs/resnet_transformer_tactile/dynamo_best_epochs10_gamma0.994_batch64_latentactiondim8
    # OUTDIR=/home/sarahp/VideoIL/runs/resnet_transformer_new_dataset/dynamo_best_epochs10_batch64_gamma0.994_latentactiondim8
    # OUTDIR=/home/sarahp/VideoIL/runs/09_17_v0.1_step1_experiments/l2_vitact_tiny_batch128_center
else
    OUTDIR=$RUNDIR/$1
fi

rm -rf $OUTDIR
python $PROJDIR/CPT/cpt/time_contrastive2.py \
    --encoder_arch vitact_tiny \
    --data_path /ssd01/gagan/cpt_data/ours/12_25_pickhuman_omnibus \
    --output_dir $OUTDIR \
    --skip_frames 10 \
    --epochs 100 \
    --batch_size_per_gpu 128 \
    --lr 5.5e-5 \
    --use_cam2    True \
    --use_cam3    True \
    --use_tactile False 
