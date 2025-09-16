RUN_ID=$(date +%Y%m%d_%H%M%S)
CONFIG=Llama-3-8B-Instruct/SFT/Llama-3-8B-Instruct.sft.full.mix_alpaca.yaml
TRAIN_FUNCTION=SFT
OUTPUT_DIR=output/$CONFIG/$RUN_ID

export PYTHONPATH=$PWD:$PYTHONPATH

python scripts/generate_weight.py --config recipes/$CONFIG.yaml --train_function $TRAIN_FUNCTION

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/train.py --config recipes/$CONFIG.yaml --output_dir $OUTPUT_DIR --train_function $TRAIN_FUNCTION

torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/test.py --config recipes/$CONFIG.yaml --output_dir $OUTPUT_DIR --train_function $TRAIN_FUNCTION