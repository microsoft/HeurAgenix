import numpy as np
from transformers import AutoModelForCausalLM
from datasets import Dataset


def get_weight(train_dataset: Dataset, **kwargs) -> np.array:
    weights = np.ones((len(train_dataset)))
    return weights