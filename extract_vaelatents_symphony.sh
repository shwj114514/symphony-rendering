export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# export CUDA_VISIBLE_DEVICES='0'

# export CUDA_VISIBLE_DEVICES='0,1,2,3'

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)



JSONL_PATH="configs/data/youtube.jsonl"
OUTPUT_DIR="./datasets/youtube"

# compute_jsonl
python  compute_latents/jsonls.py \
  --jsonl_path=${JSONL_PATH} \
  --out_dir=${OUTPUT_DIR} \
  --num_gpus ${num_gpus} \
  --augmentation_repeats=10