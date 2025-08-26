# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

import trl


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
    dataset_loader: Optional[str] = field(
        default=None,
        metadata={"help": "Dotted path to a callable returning a DatasetDict"},
    )

    dataset_process_num: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes to use for dataset processing"},
    )

    dataset_holdout_split: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the holdout split in the returned DatasetDict"},
    )

    dataset_train_split: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the train split in the returned DatasetDict"},
    )

    dataset_test_split: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the test split in the returned DatasetDict"},
    )

    weight_function: Optional[str] = field(
        default=None,
        metadata={"help": "Dotted path to a callable returning weight"},
    )

    weight_args: Optional[dict] = field(
        default=None,
        metadata={"help": "Config for weight function"},
    )

    cache_weight_file: Optional[str] = field(
        default=None,
        metadata={"help": "Dotted path to cached weight"},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})


@dataclass
class DPOConfig(trl.DPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})


@dataclass
class ORPOConfig(trl.ORPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
