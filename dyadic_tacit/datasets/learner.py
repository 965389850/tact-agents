"""Bridge processed ``dtu.dataset.v1`` rows into the directional learner.

The adapter normalizes dataset responses into a declared response catalog,
constructs causal ``Context``/``Outcome`` objects, and feeds each source
instance to two independent directional learners.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..contracts import Context, Direction, EventOrder, Outcome, stable_hash
from ..learning import DirectionalLearnerV3
from ..model import ConditionalResponseModel, parameter_hash
from ..config import default_config
from .common import direction as normalize_direction
from .common import first, load_records
from .semantics import (
    COBLOCK_RESPONSES,
    HANABI_RESPONSES,
    OVERCOOKED_CELLS,
    OVERCOOKED_RESPONSES,
    canonical_response,
    semantics_for,
    semantics_metadata,
)


DATASET_CATALOGS = {
    "overcooked": OVERCOOKED_RESPONSES,
    "hanabi": HANABI_RESPONSES,
    "coblock": COBLOCK_RESPONSES,
}
DATASET_FAMILIES = {
    "overcooked": "collab_overcooked_level4_source",
    "hanabi": "hanabi_two_player_standard",
    "coblock": "coblock_official_export",
}


def _processed_rows(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    if path.is_dir() and (path / "opportunities.jsonl").is_file():
        return load_records(path / "opportunities.jsonl")
    return load_records(path)


def _mapping_response(dataset: str, row: Mapping[str, Any]) -> str | None:
    response = row.get("response")
    if response is None:
        return None
    return canonical_response(dataset, response, row)


def _valid_responses(dataset: str, row: Mapping[str, Any], cell: str) -> tuple[str, ...]:
    catalog = DATASET_CATALOGS[dataset]
    if dataset == "overcooked":
        value = first(row, "valid_responses", "legal_responses", "legal_actions", default=[])
        # Primitive actions are converted by the observable-response extractor;
        # canonical labels are used when already present.
        if isinstance(value, str):
            value = [value]
        canonical = [str(item) for item in value if str(item) in catalog] if isinstance(value, (list, tuple)) else []
        return tuple(canonical or catalog)
    value = first(row, "valid_responses", "legal_responses", "legal_actions", "legal_commands", default=[])
    if isinstance(value, str):
        value = [value]
    values = []
    for item in value if isinstance(value, (list, tuple)) else []:
        raw = str(item).lower()
        if dataset == "hanabi":
            raw = {"reveal_color": "hint_color", "reveal_rank": "hint_rank"}.get(raw, raw)
            prefix = {
                "play": "PLAY|",
                "discard": "DISCARD|",
                "hint_color": "HINT_COLOR|",
                "hint_rank": "HINT_RANK|",
            }.get(raw)
        else:
            prefix = {
                "place": "PLACE|",
                "break": "BREAK|",
                "message": "MESSAGE|",
                "wait": "WAIT|",
            }.get(raw)
        matches = [candidate for candidate in catalog if prefix and candidate.startswith(prefix)]
        for candidate in matches:
            if candidate not in values:
                values.append(candidate)
    return tuple(values or catalog)


def _cell(dataset: str, row: Mapping[str, Any]) -> str:
    if dataset == "overcooked":
        context = row.get("context") if isinstance(row.get("context"), Mapping) else row
        return (
            OVERCOOKED_CELLS[0]
            if first(context, "handoff_state", "observable_handoff_state") is not None
            else OVERCOOKED_CELLS[1]
        )
    return "HANABI|PARTNER|NOW" if dataset == "hanabi" else "COBLOCK|PARTNER|NOW"


def _features(row: Mapping[str, Any], direction: str, dimension: int = 8) -> tuple[float, ...]:
    context = row.get("context") if isinstance(row.get("context"), Mapping) else row
    digest = stable_hash({"context": context, "direction": direction})
    return tuple(int(digest[index:index + 2], 16) / 255.0 for index in range(0, dimension * 2, 2))


def _context(dataset: str, row: Mapping[str, Any], index: int, sequence: int) -> Context:
    raw_direction = normalize_direction(row)
    self_agent, partner_agent = raw_direction.split("->")
    iid = str(first(row, "instance_id", "task_instance_id", "game_id", "episode_id", "task_id", "id"))
    cell = _cell(dataset, row)
    valid = _valid_responses(dataset, row, cell)
    if not valid:
        raise ValueError("NO_VALID_RESPONSES_FOR_LEARNER_CONTEXT")
    observation = EventOrder(index, sequence * 2)
    decision = EventOrder(index, sequence * 2 + 1)
    context_value = row.get("context") if isinstance(row.get("context"), Mapping) else row
    return Context(
        iid,
        index,
        f"{iid}:{raw_direction}:{sequence}",
        Direction(self_agent, partner_agent, DATASET_FAMILIES[dataset]),
        stable_hash({"dataset": dataset, "context": context_value, "cell": cell}),
        cell,
        _features(row, raw_direction),
        valid,
        observation,
        decision,
        self_agent,
        "own_observation",
    )


def _outcome(dataset: str, row: Mapping[str, Any], context: Context, sequence: int) -> Outcome:
    status = str(first(row, "window_status", default="complete")).lower()
    response = _mapping_response(dataset, row)
    if response is not None and response not in context.valid_responses:
        raise ValueError(f"RESPONSE_OUTSIDE_LEGAL_MASK:{response}")
    if response is None:
        reason = str(first(row, "missing_reason", "unknown_reason", default="response_not_observed"))
        status = status if status in {"unknown", "truncated", "absent"} else "unknown"
        return Outcome(
            context.instance_id,
            context.opportunity_id,
            context.direction,
            EventOrder(context.instance_index, sequence * 2 + 2),
            None,
            unknown_reason=reason,
            window_status=status,
        )
    onset = EventOrder(context.instance_index, sequence * 2 + 2)
    completion = EventOrder(context.instance_index, sequence * 2 + 2)
    return Outcome(
        context.instance_id,
        context.opportunity_id,
        context.direction,
        completion,
        response,
        onset=onset,
        completion=completion,
        window_status="complete",
    )


def _config(dataset: str, semantics) -> dict[str, Any]:
    cfg = default_config()
    cfg.update(
        # Appendix schedule: warm-up W=8 and prospective validation block
        # L_val=16.  Dataset adapters set these explicitly so they cannot
        # inherit a caller's unrelated schedule.
        warmup=8,
        block=16,
        # The paper allocates a 600-character private guidance slot.
        max_guidance_chars=600,
        context_schema_hash=stable_hash(
            {"dataset": dataset, "features": 8, "catalog": DATASET_CATALOGS[dataset]}
        ),
        screens={
            # Qualification quantities are explicit configuration values so
            # each dataset run records the thresholds it used.
            cell: dict(min_train=3, min_observed=3, min_coverage=0.8,
                       max_brier=0.30, uniform_excess=0.0)
            for cell in semantics.mapping
        },
    )
    return cfg


def train_processed_dataset(
    dataset: str,
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_split: str | None = "formation",
    seed: int = 17,
) -> dict[str, Any]:
    """Train both directional learners from processed response records.

    ``source_split`` is explicit.  The current learner's training partition is
    always formation; passing ``development`` is an explicit adapter-validation
    choice and is recorded in the result so it cannot be mistaken for the paper split.
    """
    if dataset not in DATASET_CATALOGS:
        raise ValueError(f"UNKNOWN_DATASET:{dataset}")
    rows = _processed_rows(input_path)
    if source_split is not None:
        rows = [row for row in rows if str(row.get("split", "")) == source_split]
    if not rows:
        raise ValueError(f"NO_ROWS_FOR_SPLIT:{source_split}")
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        iid = str(first(row, "instance_id", "task_instance_id", "game_id", "episode_id", "task_id", "id"))
        grouped.setdefault(iid, []).append(row)

    semantics = semantics_for(dataset)
    catalog = DATASET_CATALOGS[dataset]
    response_features = [
        [1.0 if index == position else 0.0 for index in range(len(catalog))]
        for position in range(len(catalog))
    ]
    cfg = _config(dataset, semantics)
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    learners = {
        agent: DirectionalLearnerV3(
            Direction(agent, partner, DATASET_FAMILIES[dataset]),
            ConditionalResponseModel(
                8, response_features, catalog, hidden=(128, 128), metric_dim=64,
                temperature=0.2,
            ),
            semantics,
            cfg,
            root / agent,
            seed + offset,
        )
        for offset, (agent, partner) in enumerate((("A", "B"), ("B", "A")))
    }
    before = {agent: parameter_hash(learner.train) for agent, learner in learners.items()}
    updates: list[dict[str, Any]] = []
    complete = missing = 0
    for index, (iid, instance_rows) in enumerate(grouped.items(), start=1):
        for learner in learners.values():
            learner.begin_instance(iid, index, partition="formation")
        for sequence, row in enumerate(instance_rows):
            context = _context(dataset, row, index, sequence)
            outcome = _outcome(dataset, row, context, sequence)
            learner = learners[context.direction.self_agent]
            learner.forecast(context)
            learner.observe(outcome)
            if outcome.response is None:
                missing += 1
            else:
                complete += 1
        updates.append({agent: learner.finish_instance() for agent, learner in learners.items()})
    after = {agent: parameter_hash(learner.train) for agent, learner in learners.items()}
    losses = [update["loss"] for instance_update in updates
              for update in instance_update.values() if update["loss"] is not None]
    summary = {
        "schema_version": 1,
        "status": "DATASET_LEARNER_ADAPTER_COMPLETE",
        "dataset": dataset,
        "source_split": source_split,
        "training_partition": "formation",
        "instances": len(grouped),
        "complete_responses": complete,
        "missing_responses": missing,
        "parameter_changed": {agent: before[agent] != after[agent] for agent in learners},
        "losses": losses,
        "response_catalog": list(catalog),
        "semantics": semantics_metadata(dataset),
        "updates": updates,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper learner from processed dataset records")
    parser.add_argument("dataset", choices=tuple(DATASET_CATALOGS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="formation")
    args = parser.parse_args()
    print(json.dumps(train_processed_dataset(args.dataset, args.input, args.output, source_split=args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
