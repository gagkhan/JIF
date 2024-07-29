


for gpu in 0 1; do
    export CUDA_VISIBLE_DEVICES=$gpu
    python print_cuda.py
done


