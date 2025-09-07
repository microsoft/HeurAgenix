RUN_ID=$(date +%Y%m%d_%H%M%S)
CONFIG=recipes/llama-3-8b-instruct-sft-full-weighted.yaml
OUTPUT_DIR=output/llama-3-8b-instruct-sft-full-weighted/$RUN_ID
export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/generate_weight.py --config $CONFIG
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/weighted_sft.py --config $CONFIG --output_dir $OUTPUT_DIR

torchrun torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/test.py --config recipes/llama-3-8b-instruct-sft-full-weighted.yaml --output_dir output/llama-3-8b-instruct-sft-full-weighted/20250906_200747