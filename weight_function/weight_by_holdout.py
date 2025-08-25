import numpy as np
from transformers import AutoModelForCausalLM
from datasets import Dataset


def get_weight(model: AutoModelForCausalLM, holdout_dataset: Dataset, training_dataset: Dataset, config: dict) -> np.array:
    pass