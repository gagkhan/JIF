RUNDIR="$PROJDIR/runs/toykitchen/"

if [ -z "$1" ]
    then
    echo "Output directory argument not provided"
    exit 1
else
    OUTDIR=$RUNDIR/$1
fi

gpu=2
for core in lapo ilpo ; do
    export CUDA_VISIBLE_DEVICES=$gpu 
    python $PROJDIR/CPT/cpt/main_cpt_vqvae.py \
        --encoder_arch vit_small \
        --data_path $PROJDIR/data/toykitchen/toykitchen2 \
        --output_dir $OUTDIR/core-$core \
        --beta1 0.0001 \
        --beta2 0.0001 \
        --alpha 0 \
        --core lapo \
        --batch_size_per_gpu 128 \
        --train_split 0.95 \
        --num_workers 32 \
        --warmup_epochs 0
done