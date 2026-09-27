"""Pinned upstream trained decision inference; the wrapper never repairs outputs."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from src.clients.base import (
    DecisionOption, DecisionRequest, DecisionResponse, InvalidResponseError,
    TokenUsage, UnsupportedRequestError,
)

CONTEXT_LIMIT = 8192


def public_payload(request):
    """Include both original IDs and descriptions, without scorer metadata."""
    return {
        "state": (request.state if isinstance(request.state, str) else
                  json.dumps(request.state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)),
        "question": request.question,
        "options": [json.dumps({"id": option.id, "description": option.description},
                               ensure_ascii=False, separators=(",", ":"))
                    for option in request.options],
    }


def request_from_dict(value):
    return DecisionRequest(**{**value, "options": tuple(
        DecisionOption(**option) for option in value["options"])})


def validate_result(request, result, model):
    """Keep selected-action eligibility separate from the strict score contract."""
    ids = [option.id for option in request.options]
    selected = result.get("selected_option_id")
    action = {"eligible": type(selected) is str and selected in ids,
              "selected_option_id": selected if type(selected) is str and selected in ids else None}
    try:
        if not action["eligible"]:
            raise InvalidResponseError("Invalid selected action", diagnostic_code="invalid_choice")
        probs = result.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(ids) or any(
            isinstance(p, bool) or not isinstance(p, (int, float)) or
            not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()
        ):
            raise InvalidResponseError("Invalid probabilities", diagnostic_code="invalid_probabilities")
        if abs(sum(probs.values()) - 1) > 1e-5:
            raise InvalidResponseError("Invalid probability sum", diagnostic_code="invalid_probability_sum")
        if probs[selected] < max(probs.values()) - 1e-7:
            raise InvalidResponseError("Selected action is not maximal",
                                       diagnostic_code="choice_probability_mismatch")
        logits = result.get("logits")
        if logits is not None:
            if (not isinstance(logits, dict) or set(logits) != set(ids)
                    or any(isinstance(z, bool) or not isinstance(z, (int, float))
                           or not math.isfinite(z) for z in logits.values())):
                raise InvalidResponseError("Invalid logits", diagnostic_code="invalid_probabilities")
            if logits[selected] < max(logits.values()) - 1e-7:
                raise InvalidResponseError("Logit choice mismatch",
                                           diagnostic_code="choice_probability_mismatch")
        tokens = result.get("input_tokens")
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise InvalidResponseError("Invalid usage", diagnostic_code="invalid_usage")
        response = DecisionResponse(
            request.request_id, selected, model, probabilities=probs,
            probability_kind="native", confidence=probs[selected],
            usage=TokenUsage(input_tokens=tokens, output_tokens=0, reasoning_tokens=None))
        return response, {"strict_status": "success", "action_observation": action}
    except InvalidResponseError as error:
        error.action_observation = action
        raise


def initialize(root: Path, candidate: str, device: str):
    """Load audited local safetensors and explicitly restricted head tensors."""
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if device == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight must not initialize CUDA")
    sys.path.insert(0, str(root / "upstream" / candidate))
    models = root / "models"
    if candidate == "decider":
        from decider.infer import Decider
        model = Decider(str(models / "decider"), device=device,
                        dtype=torch.bfloat16, use_graphs=False)
        temperature = model.T_by_type.get("choice", model.T)
    elif candidate == "kev":
        from kev.checkpoint import Checkpoint, Meta, LoadOptions
        from scripts.prepare_trained_jevlike import PINS
        # Bypass only the upstream constructor's implicit torch.load policy. All
        # encode, adapter loading, pointer head and forward methods are upstream.
        checkpoint = Checkpoint.__new__(Checkpoint)
        checkpoint.requested = checkpoint.path = str(models / "kev")
        meta = Meta.from_dict(torch.load(models / "kev/head.pt",
                                         map_location="cpu", weights_only=True))
        if meta.base != PINS["kev-base"][0] or meta.base_revision not in (
                None, PINS["kev-base"][1]):
            raise RuntimeError("Kev head requires a different base checkpoint")
        original_base = {"base": meta.base, "base_revision": meta.base_revision}
        meta.base = str(models / "kev-base")
        meta.base_revision = None
        checkpoint.meta = meta
        tokenizer, native = checkpoint.load(device, LoadOptions(
            dtype=torch.float32, merge=True, backend="torch",
            cuda_graphs=False, fused=False))
        model = (tokenizer, native)
        temperature = native.head.temperature
    else:
        raise ValueError("Unknown candidate")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Nonpositive/nonfinite checkpoint temperature")
    return model, {"candidate": candidate, "device": device, "temperature": temperature,
                   "context_limit": CONTEXT_LIMIT, "cpu_threads": 4,
                   "base_metadata": original_base if candidate == "kev" else None,
                   "dtype": "float32" if candidate == "kev" else "bfloat16",
                   "cuda_initialized": torch.cuda.is_initialized(),
                   "calibration_claim": False}


def infer(model, candidate, request):
    import torch
    payload = public_payload(request)
    ids = [option.id for option in request.options]
    with torch.inference_mode():
        if candidate == "decider":
            from decider.prompt import MAX_OPTIONS
            if len(ids) > MAX_OPTIONS:
                raise UnsupportedRequestError("Too many Decider options")
            # Upstream build slices context. Reject beforehand, then verify the
            # actual complete rendered length, never silently truncate.
            if len(model.m.tok.encode("Context:\n" + payload["state"],
                                      add_special_tokens=False)) > CONTEXT_LIMIT:
                raise UnsupportedRequestError("State exceeds Decider context cap")
            questions = [{"question": payload["question"], "options": payload["options"]}]
            _, items = model._decide_items([(payload["state"], questions)], CONTEXT_LIMIT)
            if len(items[0]["ids"]) > CONTEXT_LIMIT:
                raise UnsupportedRequestError("Rendered Decider request exceeds context cap")
            result = model.decide(payload["state"], questions,
                                  max_ctx_tokens=CONTEXT_LIMIT)[0]
            if result["choice"] not in payload["options"]:
                raise InvalidResponseError("Upstream choice is absent")
            selected = ids[payload["options"].index(result["choice"])]
            probs = dict(zip(ids, result["probs_list"], strict=True))
            return {"selected_option_id": selected, "probabilities": probs,
                    "input_tokens": len(items[0]["ids"]), "logits": None,
                    "upstream_choice": result["choice"]}
        tokenizer, native = model
        record = {"state": payload["state"], "questions": [
            {"instr": payload["question"], "options": payload["options"], "label": 0}]}
        enc = native.encode(tokenizer, record, max_state=CONTEXT_LIMIT,
                            max_branch=CONTEXT_LIMIT, strict=True)
        if enc["state_truncated"] or len(enc["ids"]) > CONTEXT_LIMIT:
            raise UnsupportedRequestError("Rendered Kev request exceeds context cap")
        logits = native.forward(enc)[0]
        probabilities = torch.softmax(logits, -1).cpu().tolist()
        logits_list = logits.float().cpu().tolist()
        if len(logits_list) != len(ids):
            raise InvalidResponseError("Pointer-head option count mismatch")
        selected = ids[int(logits.argmax().item())]
        return {"selected_option_id": selected,
                "probabilities": dict(zip(ids, probabilities, strict=True)),
                "logits": dict(zip(ids, logits_list, strict=True)),
                "input_tokens": len(enc["ids"])}
