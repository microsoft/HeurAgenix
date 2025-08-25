import torch
import torch.nn as nn
import numpy as np
from trl import SFTTrainer
from typing import Any, Optional, Union
from transformers import default_data_collator


class WeightedLossMixin:
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        labels  = inputs.pop("labels")
        # TODO error here.
        weights  = inputs.pop("weights")
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
    pass