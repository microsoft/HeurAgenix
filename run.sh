CUDA_VISIBLE_DEVICES=0 python eval.py -c config/MATH-500.Qwen3-8B.single.yaml -e MATH-500.Qwen3-8B.single.debug &
CUDA_VISIBLE_DEVICES=1 python eval.py -c config/MATH-500.Ministral-3-8B-Instruct-2512.single.yaml -e MATH-500.Ministral-3-8B-Instruct-2512.single.debug &
CUDA_VISIBLE_DEVICES=2 python eval.py -c config/MATH-500.Gemma2-9B-IT.single.yaml -e MATH-500.Gemma2-9B-IT.single.debug &
CUDA_VISIBLE_DEVICES=3 python eval.py -c config/MATH-500.Meta-Llama-3.1-8B-Instruct.single.yaml -e MATH-500.Meta-Llama-3.1-8B-Instruct.single.debug &