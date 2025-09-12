import torch
import torch.nn as nn
from contextlib import nullcontext
from trl import DPOTrainer
from transformers.trainer import Trainer as HFTrainer
from typing import Any, Dict, List, Literal, Optional, Union


class PairwisePreferenceCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8, max_total_length: int = 1024):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_total_length = max_total_length

    def __call__(self, features):
        feats = [f.copy() for f in features]
        example_ids = torch.tensor([f.pop("example_id") for f in feats], dtype=torch.long) \
            if len(feats) > 0 and "example_id" in feats[0] else None

        prompts   = [f["prompt"]   for f in feats]
        chosens   = [f["chosen"]   for f in feats]
        rejecteds = [f["rejected"] for f in feats]

        chosen_ans_tok   = self.tokenizer(chosens,   add_special_tokens=False, padding=False, truncation=False)
        rejected_ans_tok = self.tokenizer(rejecteds, add_special_tokens=False, padding=False, truncation=False)
        chosen_ans_lens   = torch.tensor([len(x) for x in chosen_ans_tok["input_ids"]], dtype=torch.long)
        rejected_ans_lens = torch.tensor([len(x) for x in rejected_ans_tok["input_ids"]], dtype=torch.long)

        prev_trunc_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"

        chosen_full_texts   = [p + c for p, c in zip(prompts, chosens)]
        rejected_full_texts = [p + r for p, r in zip(prompts, rejecteds)]

        chosen_batch = self.tokenizer(
            chosen_full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_total_length,
            add_special_tokens=False,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        rejected_batch = self.tokenizer(
            rejected_full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_total_length,
            add_special_tokens=False,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        self.tokenizer.truncation_side = prev_trunc_side

        batch = {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
        }

        device = batch["chosen_input_ids"].device
        Bc, Sc = batch["chosen_input_ids"].shape
        Br, Sr = batch["rejected_input_ids"].shape

        chosen_seq_len   = batch["chosen_attention_mask"].sum(dim=1)
        rejected_seq_len = batch["rejected_attention_mask"].sum(dim=1)

        chosen_ans_len_eff   = torch.minimum(chosen_ans_lens.to(device),   chosen_seq_len)
        rejected_ans_len_eff = torch.minimum(rejected_ans_lens.to(device), rejected_seq_len)

        chosen_prompt_present   = (chosen_seq_len   - chosen_ans_len_eff).clamp(min=0)
        rejected_prompt_present = (rejected_seq_len - rejected_ans_len_eff).clamp(min=0)

        ar_c = torch.arange(Sc, device=device).unsqueeze(0)
        chosen_labels = batch["chosen_input_ids"].clone()
        chosen_labels[ar_c < chosen_prompt_present.unsqueeze(1)] = -100
        chosen_labels[batch["chosen_attention_mask"] == 0] = -100

        ar_r = torch.arange(Sr, device=device).unsqueeze(0)
        rejected_labels = batch["rejected_input_ids"].clone()
        rejected_labels[ar_r < rejected_prompt_present.unsqueeze(1)] = -100
        rejected_labels[batch["rejected_attention_mask"] == 0] = -100

        batch["chosen_labels"] = chosen_labels
        batch["rejected_labels"] = rejected_labels

        if example_ids is not None:
            batch["example_id"] = example_ids
        return batch


def get_data_collator(tokenizer, max_total_length):
    return PairwisePreferenceCollator(tokenizer=tokenizer, max_total_length=max_total_length)


class WeightedDPOTrainer(DPOTrainer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.label_names = []
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device="cpu")
        self.generate_during_training = False
        self.generate_during_eval = False

    def get_batch_samples(self, dataloader_or_iter, num_samples: int = 8, device=None):
        return HFTrainer.get_batch_samples(self, dataloader_or_iter, num_samples, device)

    @torch.no_grad()
    def _gather_weights_for_batch(self, example_id: torch.Tensor, device, dtype):
        idx = example_id.detach().to("cpu").long()
        w = self.weights.index_select(0, idx)
        w = w.to(device=device, dtype=dtype)
        return w

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        example_id = batch.pop("example_id")

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        if self.args.rpo_alpha is not None:
            losses = losses * self.args.rpo_alpha + policy_nll_loss

        weights = self._gather_weights_for_batch(
            example_id,
            device=losses.device,
            dtype=losses.dtype,
        )

        loss_scalar = (weights * losses).sum() / weights.sum().clamp(min=1e-12)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().detach().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().detach().cpu()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().detach().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().detach().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()

        if self.aux_loss_enabled:
            coef = getattr(model.config, "router_aux_loss_coef", 0.0)
            loss_scalar = loss_scalar + coef * aux_loss

        return loss_scalar, metrics

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss