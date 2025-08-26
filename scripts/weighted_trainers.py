import torch
import torch.nn as nn
import numpy as np
from trl import SFTTrainer
from typing import Any, Optional, Union
from transformers import default_data_collator


class KeepKeysCollator:
    def __init__(self, base_collator, keep_key="example_id", drop_keys=("text",)):
        self.base_collator = base_collator
        self.keep_key = keep_key
        self.drop_keys = set(drop_keys) | {keep_key}

    def __call__(self, features):
        # 已经 tokenized 才能到这里
        if "input_ids" not in features[0]:
            raise RuntimeError(f"Expected tokenized features, got keys={features[0].keys()}")

        # 拿出 example_id
        ids = None
        if self.keep_key in features[0]:
            ids = torch.tensor([f[self.keep_key] for f in features], dtype=torch.long)

        # 清掉 text 和 example_id，再交给 base_collator
        cleaned = []
        for f in features:
            g = {k: v for k, v in f.items() if k not in self.drop_keys}
            cleaned.append(g)

        batch = self.base_collator(cleaned)

        # 有的 TRL 版本 collator 不会产 labels，兜底一下（可选）
        if "labels" not in batch and "input_ids" in batch:
            batch["labels"] = batch["input_ids"].clone()

        if ids is not None:
            batch[self.keep_key] = ids
        return batch




class WeightedLossMixin:
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        labels  = inputs.pop("labels")
        example_id = inputs.pop("example_id")
        # weights  = inputs.pop("weights")
        outputs = model(**inputs)
        logits  = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        bs, sl1, vocab = shift_logits.size()
        flat_logits = shift_logits.view(-1, vocab)
        flat_labels = shift_labels.view(-1)
        flat_loss   = loss_fct(flat_logits, flat_labels)
        loss_per_token = flat_loss.view(bs, sl1)

        token_mask = (shift_labels != -100).float()
        per_example_loss = (loss_per_token * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)

        weighted = per_example_loss * weights
        loss = weighted.sum() / weights.sum().clamp(min=1e-12)
        if return_outputs:
            return loss, outputs
        return loss


class WeightedSFTTrainer(WeightedLossMixin, SFTTrainer):
    def __init__(self, *args, dataset_text_field="text", **kwargs):
        self.dataset_text_field = dataset_text_field
        super().__init__(*args, dataset_text_field=dataset_text_field, **kwargs)

    def tokenize(self, examples):
        outputs = super().tokenize(examples)
        if "example_id" in examples:
            outputs["example_id"] = examples["example_id"]
        return outputs