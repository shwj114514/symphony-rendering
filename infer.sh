export CUDA_VISIBLE_DEVICES='2'
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

CKPT_PATH="checkpoints/train/transcript2muic_30s/step=350000_ema.pt"
CONFIG_PATH="./configs/transcript2muic_30s.yaml"


python infer.py \
    --config=${CONFIG_PATH} \
    --ckpt ${CKPT_PATH}

# CKPT_PATH="checkpoints/train/transcript2muic_10s/step=150000_ema.pt"
# CONFIG_PATH="./configs/transcript2muic_10s.yaml"


# ${PYTHON_PATH} infer.py \
#     --config=${CONFIG_PATH} \
#     --ckpt ${CKPT_PATH}