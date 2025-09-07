RUN_ID=$(date +%Y%m%d_%H%M%S)
CONFIG=llama-3-8b-instruct-sft-full.mix_alpaca
OUTPUT_DIR=output/$CONFIG/$RUN_ID

export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/generate_weight.py --config recipes/$CONFIG.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/weighted_sft.py --config recipes/$CONFIG.yaml --output_dir $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/test.py --config recipes/$CONFIG.yaml --output_dir $OUTPUT_DIR