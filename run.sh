RUN_ID=$(date +%Y%m%d_%H%M%S)
CONFIG=recipes/llama-3-8b-instruct-sft-full-weighted.yaml
OUTPUT_DIR=output/llama-3-8b-instruct-sft-full-weighted/$RUN_ID
export PYTHONPATH=$PWD:$PYTHONPATH

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/weighted_sft.py --config $CONFIG --output_dir $OUTPUT_DIR

torchrun --nproc_per_node=4 scripts/test.py --config $CONFIG --output_dir $OUTPUT_DIR