"""Single-forward SemIf scoring for a small batch of short decisions."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any

SEMIF_SNAKE_PROMPT_VERSION = "snake-options-v1"


def encode_snake_prompt(
    tokenizer: Any, row: dict[str, Any], max_tokens: int
) -> tuple[list[int], list[int], str]:
    """Encode the same SemIf decision contract with a compact Snake-specific prompt."""

    from semif_phase1.core import LETTERS, digest, validate_row

    validate_row(row)
    options = "\n".join(
        f"{LETTERS[index]}: {option['description']}"
        for index, option in enumerate(row["options"])
    )
    message = (
        "Choose exactly one option for the state and question. Reply only with its letter.\n"
        f"State: {row['state']}\nQuestion: {row['question']}\n{options}"
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(
            f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; "
            "no truncation allowed"
        )
    slots = []
    for letter in LETTERS[: len(row["options"])]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        token = encoded[0]
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
        slots.append(token)
    return ids, slots, digest(prompt)


def score_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    max_tokens: int = 4096,
    encoder: Callable[
        [Any, dict[str, Any], int], tuple[list[int], list[int], str]
    ]
    | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score independent SemIf rows in one padded forward pass.

    SemIf's shared-prefix path is ideal for many questions over a long state. Snake has only three
    short questions, so recomputing the tiny common prefix in one batch is cheaper than a separate
    prefix prefill followed by a cached suffix pass.
    """

    import torch
    from semif_phase1.core import softmax
    from semif_phase1.direct import PROMPT_VERSION, encode_prompt

    if not rows:
        raise ValueError("SemIf batch requires at least one decision")
    started = time.perf_counter()
    selected_encoder = encoder or encode_prompt
    prompt_version = SEMIF_SNAKE_PROMPT_VERSION if encoder else PROMPT_VERSION
    encoded = [selected_encoder(tokenizer, row, max_tokens) for row in rows]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer requires a padding or EOS token")
    width = max(len(ids) for ids, _, _ in encoded)
    ends = [len(ids) - 1 for ids, _, _ in encoded]
    selected_positions = sorted(set(ends))
    input_ids = []
    attention_masks = []
    position_ids = []
    for ids, _, _ in encoded:
        padding = width - len(ids)
        input_ids.append(ids + [pad] * padding)
        attention_masks.append([1] * len(ids) + [0] * padding)
        position_ids.append(list(range(len(ids))) + [0] * padding)
    encode_seconds = time.perf_counter() - started

    device = next(model.parameters()).device
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" not in parameters and hasattr(model, "get_base_model"):
        parameters = inspect.signature(model.get_base_model().forward).parameters
    if "logits_to_keep" not in parameters:
        raise RuntimeError("Model lacks selective-position logits needed by batched scoring")

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    inputs = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
        "position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
    }
    sync()
    forward_started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            **inputs,
            use_cache=False,
            return_dict=True,
            logits_to_keep=torch.tensor(selected_positions, dtype=torch.long, device=device),
        )
    sync()
    forward_seconds = time.perf_counter() - forward_started

    results = []
    for index, (row, (ids, slots, prompt_hash)) in enumerate(zip(rows, encoded)):
        vocabulary = output.logits[index, selected_positions.index(ends[index]), :].float()
        selected = vocabulary[slots].cpu().tolist()
        results.append(
            {
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "probabilities": softmax(selected),
                "option_logits": selected,
                "input_tokens": len(ids),
                "prompt_sha256": prompt_hash,
                "prompt_version": prompt_version,
                "model": {**metadata, "serving_config": "single-forward-batch-v1"},
                "readout": "native selected last-position logits from one padded batch",
                "probability_status": (
                    "conditional option score; uncalibrated as decision confidence"
                ),
            }
        )
    del output
    return results, {
        "total_seconds": time.perf_counter() - started,
        "encode_seconds": encode_seconds,
        "forward_seconds": forward_seconds,
        "batch_size": len(rows),
        "true_tokens": sum(len(ids) for ids, _, _ in encoded),
        "padded_tokens": len(rows) * width,
    }
