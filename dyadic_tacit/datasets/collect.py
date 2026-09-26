"""Collect observable response records from frozen LLM episode snapshots.

The original Overcooked host obtains a public snapshot, asks the frozen model
for one legal partner action, advances the environment, and records the
observable result.  Hanabi and CoBlock expose different environment objects,
but use the same data contract here: one input row is one pre-decision public
snapshot and one provider call supplies the partner response.  An environment
adapter can replace the snapshot source without changing provider handling or
the downstream processors.

This module does not store raw model text or hidden reasoning.  It stores only
the parsed legal response and safe provider accounting metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..providers import FrozenLLM, ProviderReply, provider_from_environment
from .common import first, load_records


DATASETS = ("hanabi", "coblock")


def _legal(row: Mapping[str, Any]) -> list[str]:
    value = first(row, "valid_responses", "legal_responses", "legal_actions", "legal_commands", default=[])
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item)]


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _choice(text: str, legal: Sequence[str]) -> str:
    """Extract one legal choice from a provider's constrained response."""
    if not legal:
        raise ValueError("NO_LEGAL_RESPONSES_IN_SNAPSHOT")
    candidates: list[str] = []
    try:
        decoded = json.loads(text)
        if isinstance(decoded, Mapping):
            for key in ("response", "action", "action_type", "command", "choice"):
                value = decoded.get(key)
                if value is not None:
                    candidates.append(str(value))
    except (TypeError, json.JSONDecodeError):
        pass
    candidates.append(str(text))
    normalised_legal = {_normalise(item): item for item in legal}
    aliases = {
        "hint_colour": "hint_color",
        "hint_colour_action": "hint_color",
        "hint_color_action": "hint_color",
    }
    for candidate in candidates:
        key = aliases.get(_normalise(candidate), _normalise(candidate))
        if key in normalised_legal:
            return normalised_legal[key]
    lowered = _normalise(text)
    for key, original in sorted(normalised_legal.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            return original
    raise ValueError("PROVIDER_RESPONSE_OUTSIDE_LEGAL_MASK")


def _decoded(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _prompt(dataset: str, row: Mapping[str, Any], legal: Sequence[str]) -> list[dict[str, str]]:
    context = row.get("context") if isinstance(row.get("context"), Mapping) else row
    safe_context = {str(k): v for k, v in context.items() if not str(k).startswith("_")}
    return [
        {
            "role": "system",
            "content": (
                "You are a frozen cooperative partner in the " + dataset +
                " environment. Use only the public observation. Return one JSON object "
                "with the field response and no explanation. For Hanabi also include "
                "card_position or hint_referent when applicable; for CoBlock include "
                "block_or_region_target when applicable and include timing when observable."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"public_context": safe_context, "legal_responses": list(legal)},
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def _response(dataset: str, action: str, text: str, row: Mapping[str, Any]) -> dict[str, Any]:
    decoded = _decoded(text)
    timing = decoded.get("timing", first(row, "response_timing", "timing"))
    if dataset == "hanabi":
        result: dict[str, Any] = {"action_type": action}
        card_position = decoded.get(
            "card_position", first(row, "card_position", "card_index", default=None)
        )
        hint_referent = decoded.get(
            "hint_referent", first(row, "hint_referent", "hint_target", default=None)
        )
        if card_position is not None:
            result["card_position"] = card_position
        if hint_referent is not None:
            result["hint_referent"] = hint_referent
        if timing is not None:
            result["timing"] = timing
        return result
    if dataset == "coblock":
        result = {"action": action}
        target = decoded.get(
            "block_or_region_target",
            first(row, "block_or_region_target", "block_target", "region_target", "target", default=None),
        )
        if target is not None:
            result["block_or_region_target"] = target
        if timing is not None:
            result["timing"] = timing
        return result
    raise ValueError(f"UNSUPPORTED_COLLECTION_DATASET:{dataset}")


def _provider_metadata(reply: ProviderReply) -> dict[str, Any]:
    return {
        "provider": reply.provider,
        "model": reply.model,
        "request_id": reply.request_id,
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
    }


def collect_snapshots(
    dataset: str,
    input_path: str | Path,
    output_path: str | Path,
    provider: FrozenLLM,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Collect complete response windows from a public snapshot stream.

    The input may be a JSON/JSONL file or directory accepted by the dataset
    processors.  The output is JSONL in the same row shape, ready for
    ``dyadic_tacit.datasets.process``.
    """
    if dataset not in DATASETS:
        raise ValueError(f"COLLECTION_REQUIRES_ONE_OF:{','.join(DATASETS)}")
    rows = load_records(input_path)
    if not rows:
        raise ValueError("EMPTY_COLLECTION_INPUT")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = missing = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            item = dict(row)
            legal = _legal(item)
            try:
                reply = provider.complete(_prompt(dataset, item, legal), role="partner")
                action = _choice(reply.text, legal)
                item["response"] = _response(dataset, action, reply.text, item)
                item["window_status"] = "complete"
                item["response_window"] = {"status": "complete", "steps": 1}
                item["collection_status"] = "complete"
                # The generic collector is the host boundary.  With the
                # snapshot source used here, the selected legal action is
                # recorded as a completed replay decision; a native HLE or
                # CoBlock executor can replace this field with its transition
                # acknowledgement without changing the downstream schema.
                item["execution_status"] = "snapshot_replay"
                item["collection_step"] = index
                item["provider_metadata"] = _provider_metadata(reply)
                completed += 1
            except Exception as exc:
                if not allow_missing:
                    raise
                item["response"] = None
                item["window_status"] = "unknown"
                item["missing_reason"] = f"collection_failed:{type(exc).__name__}"
                item["collection_status"] = "missing"
                item["collection_step"] = index
                missing += 1
            handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    return {
        "status": "RESPONSE_COLLECTION_COMPLETE" if not missing else "RESPONSE_COLLECTION_PARTIAL",
        "dataset": dataset,
        "input": str(input_path),
        "output": str(output),
        "records": len(rows),
        "complete_responses": completed,
        "missing_responses": missing,
        "provider": {"name": provider.provider, "model": provider.model},
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Collect provider responses for Hanabi or CoBlock snapshots")
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=("gpt", "claude", "gemini"), default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    provider: FrozenLLM = provider_from_environment(args.provider)
    result = collect_snapshots(
        args.dataset,
        args.input,
        args.output,
        provider,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
