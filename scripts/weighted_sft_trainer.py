import torch
import torch.nn as nn
from typing import Any, Optional, Union
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from alignment.dataset_utils import infer_response_template


class KeepKeysWrapper:
    def __init__(self, base_collator, keep_key="example_id"):
        self.base_collator = base_collator
        self.keep_key = keep_key

    def __call__(self, features):
        feats = [f.copy() for f in features]

        ids = None
        if len(feats) > 0 and self.keep_key in feats[0]:
            ids = torch.tensor([f.pop(self.keep_key) for f in feats], dtype=torch.long)

        already_tokenized = len(feats) > 0 and ("input_ids" in feats[0] or "labels" in feats[0])
        if already_tokenized:
            for f in feats:
                f.pop("text", None)
                f.pop("message", None)

        batch = self.base_collator(feats)

        if ids is not None:
            batch[self.keep_key] = ids
        return batch

    def __getattr__(self, name):
        try:
            return getattr(self.base_collator, name)
        except AttributeError:
            raise


class EoTCompletionCollator:
    def __init__(self, base_collator, tokenizer):
        self.base = base_collator
        self.eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def __call__(self, features):
        batch = self.base(features)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        B, S = labels.shape
        for i in range(B):
            sup = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if sup.numel() == 0:
                continue
            last = sup[-1].item()
            if last + 1 < S and input_ids[i, last + 1].item() == self.eot_id:
                labels[i, last + 1] = self.eot_id
        batch["labels"] = labels
        return batch

    def __getattr__(self, name):
        try:
            return getattr(self.base, name)
        except AttributeError:
            raise


def get_data_collator(tokenizer):
    response_template, response_template_ids = infer_response_template(tokenizer)
    try:
        base= DataCollatorForCompletionOnlyLM(response_template_ids=response_template_ids, tokenizer=tokenizer)
    except TypeError:
        base= DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    base = EoTCompletionCollator(base, tokenizer)
    data_collator = KeepKeysWrapper(base_collator=base, keep_key="example_id")
    return data_collator


class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.label_names = []
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def _gather_weights_for_batch(self, example_id: torch.Tensor, device, dtype):
        idx = example_id.detach().to("cpu").long()
        weight = self.weights.index_select(0, idx)
        weight = weight.to(device=device, dtype=dtype)
        return weight

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
    