import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
from trl import ModelConfig, TrlParser, DPOConfig


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class WeightConfig:
    """Configuration for a weight calculation."""
    weight_function: callable
    weight_normalization_function: callable
    cache_file: str = None


@dataclass
class DataConfig:
    dataset_loader: Optional[str] = field(default=None, metadata={"help": "Dotted path to a callable returning a DatasetDict"})
    dataset_process_num: Optional[int] = field(default=None, metadata={"help": "Number of processes to use for dataset processing"})
    dataset_holdout_split: Optional[str] = field(default=None, metadata={"help": "Name of the holdout split in the returned DatasetDict"})
    dataset_train_split: Optional[str] = field(default=None, metadata={"help": "Name of the train split in the returned DatasetDict"})
    dataset_test_split: Optional[str] = field(default=None, metadata={"help": "Name of the test split in the returned DatasetDict"})
    weight_function: Optional[str] = field(default=None, metadata={"help": "Dotted path to a callable returning weight"})
    weight_args: Optional[dict] = field(default_factory=dict, metadata={"help": "Config for weight function"}, )


@dataclass
class TestConfig:
    eval_function: Optional[str] = field(default=None, metadata={"help": "The evaluation function to use."})
    eval_args: Optional[dict] = field(default_factory=dict, metadata={"help": "The evaluation arguments to use."})


@dataclass
class SimPOConfig(DPOConfig):
    beta: float = 2.0
    gamma_beta_ratio: float = 0.25
    sft_weight: float = 0.0
    label_smoothing: float = 0
    loss_type: Literal["sigmoid", "hinge"] = "sigmoid"
    disable_dropout: bool = True

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--train_function", "--tf", type=str, default="SFT", choices=["SFT", "DPO", "SimPO"], help="Specify the training function.")
    tf_args, remaining = pre.parse_known_args()

    train_function = (tf_args.train_function or "").strip().upper()
    if train_function == "SFT":
        from trl import SFTConfig as TrainConfig
        train_function = "SFT"
    elif train_function == "DPO":
        from trl import DPOConfig as TrainConfig
        train_function = "DPO"
    elif train_function == "SIMPO":
        from alignment.configs import SimPOConfig as TrainConfig
        train_function = "SimPO"
    sys.argv = [sys.argv[0]] + remaining
    parser = TrlParser((ModelConfig, DataConfig, TrainConfig, TestConfig))
    model_args, data_args, training_args, test_args = parser.parse_args_and_config()
    return model_args, data_args, training_args, test_args, train_function