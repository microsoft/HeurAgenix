import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, PreTrainedTokenizerBase


class WeightedSimPOTrainer(Trainer):
    def __init__(
        self,
        weights: Union[List[float], torch.Tensor],
        beta: float,
        gamma_beta_ratio: float,
        loss_type: str = "sigmoid",
        sft_weight: float = 0.0,
        weight_sft: bool = False,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.beta = beta
        self.gamma_beta_ratio = gamma_beta_ratio
        self.loss_type = loss_type
        self.sft_weight = sft_weight
        self.weight_sft = weight_sft
        self.tokenizer = tokenizer

        self.label_pad_token_id = self.args.label_pad_token_id
        self.padding_value = self.args.padding_value
        if self.padding_value is None and tokenizer is not None:
            self.padding_value = tokenizer.pad_token_id

        self.weights = torch.as_tensor(weights, dtype=torch.float32, device="cpu")

        if self.args.disable_dropout:
            self._disable_dropout_in_model(self.model)

    @staticmethod
    def _disable_dropout_in_model(model: nn.Module):
        for module in model.modules():
            if isinstance(module, (nn.Dropout,)):
                module.p = 0.0

    @torch.no_grad()
    def _gather_weights_for_batch(self, example_id: Optional[torch.Tensor], device, dtype):
        if example_id is None:
            return None
        idx = example_id.detach().to("cpu").long()
        w = self.weights.index_select(0, idx).to(device=device, dtype=dtype)
        return w

    @staticmethod
    def _concatenate_inputs(
        batch: Dict[str, torch.Tensor],
        is_encoder_decoder: bool,
        label_pad_token_id: int,
        padding_value: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        def _pad_to_length(x: torch.Tensor, tgt_len: int, pad_value: int):
            if x.size(1) >= tgt_len:
                return x
            pad_amt = tgt_len - x.size(1)
            return F.pad(x, (0, pad_amt), value=pad_value)

        assert not is_encoder_decoder, "This implementation targets decoder-only models."

        chosen_input_ids = batch["chosen_input_ids"].to(device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)

        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        max_len = max(chosen_input_ids.size(1), rejected_input_ids.size(1))
        chosen_input_ids = _pad_to_length(chosen_input_ids, max_len, pad_value=padding_value)
        chosen_attention_mask = _pad_to_length(chosen_attention_mask, max_len, pad_value=0)
        chosen_labels = _pad_to_length(chosen_labels, max_len, pad_value=label_pad_token_id)

        rejected_input_ids = _pad_to_length(rejected_input_ids, max_len, pad_value=padding_value)
        rejected_attention_mask = _pad_to_length(rejected_attention_mask, max_len, pad_value=0)
        rejected_labels = _pad_to_length(rejected_labels, max_len, pad_value=label_pad_token_id)

        concatenated = {
            "concatenated_input_ids": torch.cat([chosen_input_ids, rejected_input_ids], dim=0),
            "concatenated_attention_mask": torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0),
            "concatenated_labels": torch.cat([chosen_labels, rejected_labels], dim=0),
        }
        return concatenated

    @staticmethod
    def _get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Shape mismatch between logits and labels.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        safe_labels = labels.clone()
        safe_labels[~loss_mask] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _simpo_loss(self, avg_logp_chosen: torch.Tensor, avg_logp_rejected: torch.Tensor):
        logits = avg_logp_chosen - avg_logp_rejected - self.gamma_beta_ratio
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(f"Unsupported loss_type={self.loss_type}")
        chosen_rewards = (self.beta * avg_logp_chosen).detach()
        rejected_rewards = (self.beta * avg_logp_rejected).detach()
        return losses, chosen_rewards, rejected_rewards

    def _concatenated_forward(self, model: nn.Module, batch: Dict[str, torch.Tensor]):
        device = self.accelerator.device if hasattr(self, "accelerator") else self.args.device
        concat = self._concatenate_inputs(
            batch,
            is_encoder_decoder=False,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=device,
        )
        out = model(
            concat["concatenated_input_ids"],
            attention_mask=concat["concatenated_attention_mask"],
            use_cache=False,
        )
        logits = out.logits
        logps_sum = self._get_batch_logps(
            logits, concat["concatenated_labels"],
            label_pad_token_id=self.label_pad_token_id,
            is_encoder_decoder=False,
            average_log_prob=False,
        )
        len_chosen = batch["chosen_labels"].shape[0]
        chosen_sum = logps_sum[:len_chosen]
        rejected_sum = logps_sum[len_chosen:]
        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen:]
        chosen_labels = concat["concatenated_labels"][:len_chosen]
        return chosen_sum, rejected_sum, chosen_logits, rejected_logits, chosen_labels

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ):
        example_id = inputs.pop("example_id", None)
        chosen_sum, rejected_sum, chosen_logits, rejected_logits, chosen_labels = self._concatenated_forward(model, inputs)

        chosen_mask = (inputs["chosen_labels"] != self.label_pad_token_id)
        rejected_mask = (inputs["rejected_labels"] != self.label_pad_token_id)
        chosen_len = chosen_mask.sum(dim=1).clamp(min=1).to(chosen_sum.dtype)
        rejected_len = rejected_mask.sum(dim=1).clamp(min=1).to(rejected_sum.dtype)
        avg_chosen = chosen_sum / chosen_len
        avg_rejected = rejected_sum / rejected_len

        losses, chosen_rewards, rejected_rewards = self._simpo_loss(avg_chosen, avg_rejected)

        weights = self._gather_weights_for_batch(example_id, device=losses.device, dtype=losses.dtype)
        loss_scalar = (weights * losses).sum() / weights.sum().clamp(min=1e-12)

        if self.sft_weight and self.sft_weight > 0.0:
            logits_shifted = chosen_logits[..., :-1, :].contiguous()
            labels_shifted = inputs["chosen_labels"][..., 1:].clone().to(logits_shifted.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id, reduction="none")
            ce_per_token = loss_fct(
                logits_shifted.reshape(-1, logits_shifted.size(-1)),
                labels_shifted.reshape(-1),
            ).view(labels_shifted.shape[0], -1)
            valid_mask = (labels_shifted != self.label_pad_token_id)
            ce_sum = (ce_per_token * valid_mask).sum(dim=1)
            ce_cnt = valid_mask.sum(dim=1).clamp(min=1)
            sft_loss_per_ex = ce_sum / ce_cnt
            if self.weight_sft:
                sft_loss = (weights * sft_loss_per_ex).sum() / weights.sum().clamp(min=1e-12)
            else:
                sft_loss = sft_loss_per_ex.mean()
            loss_scalar = loss_scalar + self.sft_weight * sft_loss

        with torch.no_grad():
            logs = {
                "rewards/chosen": chosen_rewards.mean().detach().cpu(),
                "rewards/rejected": rejected_rewards.mean().detach().cpu(),
                "rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean().detach().cpu(),
                "rewards/margins": (chosen_rewards - rejected_rewards).mean().detach().cpu(),
                "logps/chosen": avg_chosen.detach().mean().cpu(),
                "logps/rejected": avg_rejected.detach().mean().cpu(),
                "logits/chosen": chosen_logits.detach().mean().cpu(),
                "logits/rejected": rejected_logits.detach().mean().cpu(),
            }
            self.log({k: float(v) for k, v in logs.items()})

        return (loss_scalar, None) if return_outputs else loss_scalar