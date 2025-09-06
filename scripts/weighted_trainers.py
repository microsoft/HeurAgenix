import torch
import torch.nn as nn
from typing import Any, Optional, Union
from trl import SFTTrainer


class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.label_names = []

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

        weighted = per_example_loss * weights
        loss = weighted.sum() / weights.sum().clamp(min=1e-12)
        if return_outputs:
            return loss, outputs
        return loss
    