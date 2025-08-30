import torch
import torch.nn as nn
import numpy as np
from trl import SFTTrainer
from typing import Any, Optional, Union


class KeepKeysCollator:
    def __init__(self, base_collator, keep_key="example_id", drop_keys=("text", "message")):
        self.base_collator = base_collator
        self.keep_key = keep_key
        self.drop_keys = set(drop_keys) | {keep_key}

    def __call__(self, features):
        if "input_ids" not in features[0]:
            raise RuntimeError(f"Expected tokenized features, got keys={features[0].keys()}")

        ids = None
        if self.keep_key in features[0]:
            ids = torch.tensor([f[self.keep_key] for f in features], dtype=torch.long)

        cleaned = []
        for f in features:
            g = {k: v for k, v in f.items() if k not in self.drop_keys}
            cleaned.append(g)

        batch = self.base_collator(cleaned)

        if "labels" not in batch and "input_ids" in batch:
            batch["labels"] = batch["input_ids"].clone()

        if ids is not None:
            batch[self.keep_key] = ids
        return batch


class WeightedLossMixin:
    # TODO: dynamic update
    # def __init__(self, model, weight_function, holdout_dataset, train_dataset, weight_args, **kwargs):
    def __init__(self, weights, **kwargs):
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def _gather_weights_for_batch(self, example_id: torch.Tensor, device, dtype):
        idx = example_id.detach().to("cpu").long()
        w = self.weights.index_select(0, idx)
        w = w.to(device=device, dtype=dtype)
        return w

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        labels  = inputs.pop("labels")
        example_id = inputs.pop("example_id")
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
        weights = self._gather_weights_for_batch(
            example_id,
            device=per_example_loss.device,
            dtype=per_example_loss.dtype,
        )

        weighted = per_example_loss # * weights
        loss = weighted.sum() / weights.sum().clamp(min=1e-12)
        if return_outputs:
            return loss, outputs
        return loss


class WeightedSFTTrainer(SFTTrainer, WeightedLossMixin):
    def __init__(self, *args, weight_function, holdout_dataset, weight_args, weights, **kwargs):
        SFTTrainer.__init__(self, *args, **kwargs)
        WeightedLossMixin.__init__(
            self,
            # model=self.model,
            # weight_function=weight_function,
            # holdout_dataset=holdout_dataset,
            # train_dataset=self.train_dataset,
            # weight_args=weight_args,
            weights=weights
        )
        self.label_names = []
        self.data_collator = KeepKeysCollator(self.data_collator, keep_key="example_id", drop_keys=("text", "message"))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return WeightedLossMixin.compute_loss(
            self, model, inputs, return_outputs, num_items_in_batch
        )

    def tokenize(self, examples):
        outputs = super().tokenize(examples)
        if "example_id" in examples:
            outputs["example_id"] = examples["example_id"]
        return outputs