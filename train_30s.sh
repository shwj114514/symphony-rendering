export CUDA_VISIBLE_DEVICES='5'
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python train.py \
    --config="./configs/transcript2muic_30s.yaml" \
    --no_log
