"""Shared, lossless utilities for the three paper dataset adapters.

The adapters intentionally do not fabricate labels or favorable snapshots.  A
source row that has no response is retained with an explicit missing status,
so coverage denominators remain visible in later audit code.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SPLITS = {"development", "formation", "main_audit", "hard_audit", "utility"}
_SPLIT_ALIASES = {
    "dev": "development", "development": "development",
    "train": "formation", "formation": "formation",
    "main": "main_audit", "main_audit": "main_audit",
    "hard": "hard_audit", "hard_audit": "hard_audit",
    "util": "utility", "utility": "utility",
}
_DIRECTION_ALIASES = {
    "a->b": "A->B", "a_to_b": "A->B", "a-b": "A->B", "0->1": "A->B",
    "b->a": "B->A", "b_to_a": "B->A", "b-a": "B->A", "1->0": "B->A",
}


def first(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None and row[name] != "":
            return row[name]
    return default


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL, or CSV files, recursively when *path* is a folder."""
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    supported = {".json", ".jsonl", ".ndjson", ".csv"}
    files = [p for p in files if p.suffix.lower() in supported]
    if not files:
        raise ValueError(f"no JSON/JSONL/CSV input files found at {root}")
    out: list[dict[str, Any]] = []
    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".csv":
            with file.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        else:
            text = file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            if suffix == ".json":
                parsed = json.loads(text)
                if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
                    rows = parsed["records"]
                elif isinstance(parsed, list):
                    rows = parsed
                else:
                    rows = [parsed]
            else:
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"{file}:{row_index + 1} is not an object")
            item = dict(row)
            item["_source_file"] = str(file)
            item["_source_row"] = row_index
            out.append(item)
    return out


def expand_opportunities(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand optional opportunities/snapshots arrays while retaining metadata."""
    expanded: list[dict[str, Any]] = []
    for parent in rows:
        child_key = next((key for key in ("opportunities", "snapshots", "probe_rows") if isinstance(parent.get(key), list)), None)
        if child_key is None:
            expanded.append(dict(parent))
            continue
        for index, child in enumerate(parent[child_key]):
            if not isinstance(child, Mapping):
                raise ValueError(f"{parent.get('_source_file')} opportunity {index} is not an object")
            merged = dict(parent)
            merged.pop(child_key, None)
            merged.update(dict(child))
            merged["_source_parent"] = parent.get("_source_row")
            merged["_source_row"] = f"{parent.get('_source_row')}:{index}"
            expanded.append(merged)
    return expanded


def load_mapping(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("mapping files must contain a JSON object")
    return parsed


def load_split_manifest(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = load_records(path)
    manifest: dict[str, str] = {}
    for row in rows:
        instance_id = str(first(row, "instance_id", "task_instance_id", "game_id", "episode_id", "task_id", "id", default=""))
        if not instance_id:
            raise ValueError("split manifest row has no instance_id")
        split = normalize_split(first(row, "split", "partition"))
        if instance_id in manifest and manifest[instance_id] != split:
            raise ValueError(f"conflicting split assignments for {instance_id}")
        manifest[instance_id] = split
    return manifest


def normalize_split(value: Any) -> str:
    key = str(value or "").strip().lower().replace(" ", "_")
    if key not in _SPLIT_ALIASES:
        raise ValueError(f"unknown split {value!r}; use one of {sorted(SPLITS)}")
    return _SPLIT_ALIASES[key]


def resolve_split(row: Mapping[str, Any], manifest: Mapping[str, str], instance_id: str) -> str:
    from_row = first(row, "split", "partition")
    from_manifest = manifest.get(instance_id)
    if from_row is None and from_manifest is None:
        raise ValueError(f"{instance_id}: split is required (source field or --split-manifest)")
    if from_row is not None and from_manifest is not None and normalize_split(from_row) != from_manifest:
        raise ValueError(f"{instance_id}: source split disagrees with split manifest")
    return from_manifest or normalize_split(from_row)


def instance_id(row: Mapping[str, Any]) -> str:
    value = first(row, "instance_id", "task_instance_id", "game_id", "episode_id", "task_id", "id")
    if value is None:
        raise ValueError("source row has no instance_id/game_id/episode_id/task_id/id")
    return str(value)


def direction(row: Mapping[str, Any]) -> str:
    value = first(row, "direction", "query_direction", "self_direction")
    if value is None:
        self_agent = str(first(row, "self_agent", "agent", "role_id", default="")).lower()
        value = "A->B" if self_agent in {"a", "0", "agent_0", "player_0"} else "B->A" if self_agent in {"b", "1", "agent_1", "player_1"} else None
    if value is None:
        raise ValueError("source row has no direction or self_agent")
    key = str(value).strip().lower()
    if key not in _DIRECTION_ALIASES:
        raise ValueError(f"unknown direction {value!r}; use A->B or B->A")
    return _DIRECTION_ALIASES[key]


def clean_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean_for_json(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(v) for v in value]
    return value


def drop_keys(value: Any, forbidden: set[str]) -> Any:
    """Recursively remove known hidden-information fields from context."""
    if isinstance(value, Mapping):
        return {str(k): drop_keys(v, forbidden) for k, v in value.items() if str(k).lower() not in forbidden}
    if isinstance(value, list):
        return [drop_keys(v, forbidden) for v in value]
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(clean_for_json(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def response_window(row: Mapping[str, Any], response: Any) -> tuple[Any, str, str | None]:
    status = str(first(row, "window_status", "response_status", default="")).lower().strip()
    if status not in {"complete", "unknown", "truncated", "absent"}:
        status = "complete" if response is not None else "absent"
    if response is None and status == "complete":
        status = "absent"
    reason = first(row, "missing_reason", "unknown_reason")
    return response, status, None if reason is None else str(reason)


def usage_value(row: Mapping[str, Any], response: Mapping[str, Any] | None, mapping: Mapping[str, Any], dataset: str) -> Any:
    direct = first(row, "usage", "usage_label")
    if direct is not None:
        return direct
    if response is None:
        return None
    # Some exports store the observed usage beside the action inside the
    # response object. Preserve that value before consulting any optional
    # external mapping; never invent a label from the action name.
    nested = first(response, "usage", "usage_label", "usage_id")
    if nested is not None:
        return nested
    key = str(first(response, "action_type", "action", "relation", "type", default=""))
    scoped = mapping.get(dataset, mapping)
    if isinstance(scoped, Mapping):
        return scoped.get(key)
    return None


def build_record(
    *, dataset: str, row: Mapping[str, Any], family_id: str, split: str,
    context: Mapping[str, Any], valid_responses: Iterable[Any], response: Mapping[str, Any] | None,
    usage_mapping: Mapping[str, Any], warnings: list[str], forbidden_context_keys: set[str] | None = None,
) -> dict[str, Any]:
    context_clean = clean_for_json(context)
    if forbidden_context_keys:
        context_clean = drop_keys(context_clean, {k.lower() for k in forbidden_context_keys})
    response_value, window_status, missing_reason = response_window(row, response)
    if response_value is None and window_status in {"unknown", "truncated"} and missing_reason is None:
        missing_reason = "response_not_observed"
    canonical_response = None
    usage = usage_value(row, response_value, usage_mapping, dataset)
    semantic_hash = None
    if response_value is not None:
        # The appendix requires complete C to be retained while qualification
        # and guidance use its fixed U projection.  Keep this import lazy so
        # the generic JSON/CSV utilities remain usable independently.
        from .semantics import semantic_usage, semantics_for
        canonical_response, semantic_usage_value = semantic_usage(dataset, response_value, row)
        if usage is None:
            usage = semantic_usage_value
        semantic_hash = semantics_for(dataset).schema_hash()
    if response_value is not None and usage is None:
        warnings.append("complete_response_has_no_usage_mapping")
    source = {"file": row.get("_source_file"), "row": row.get("_source_row")}
    return {
        "schema_version": "dtu.dataset.v1",
        "dataset": dataset,
        "family_id": family_id,
        "instance_id": instance_id(row),
        "split": split,
        "direction": direction(row),
        "context": context_clean,
        "context_features": context_clean,
        "context_key": stable_hash(context_clean),
        "verification_cell": stable_hash({"family_id": family_id, "context": context_clean}),
        "valid_responses": list(valid_responses),
        "response": response_value,
        "response_canonical": canonical_response,
        "response_schema_version": "dtu.response.v1",
        "response_window": {"status": window_status, "missing_reason": missing_reason},
        "window_status": window_status,
        "missing_reason": missing_reason,
        "usage": usage,
        "usage_map": usage,
        "usage_schema_version": semantic_hash,
        "warnings": sorted(set(warnings)),
        "source": source,
        "source_hash": stable_hash(row),
    }


def write_processed(records: list[dict[str, Any]], output: str | Path) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    opportunities = root / "opportunities.jsonl"
    with opportunities.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    instances: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["instance_id"]
        current = instances.setdefault(key, {
            "schema_version": "dtu.dataset.v1", "dataset": record["dataset"],
            "family_id": record["family_id"], "instance_id": key,
            "split": record["split"], "directions": [],
        })
        if record["direction"] not in current["directions"]:
            current["directions"].append(record["direction"])
    with (root / "instances.jsonl").open("w", encoding="utf-8") as handle:
        for item in sorted(instances.values(), key=lambda x: x["instance_id"]):
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "dtu.dataset.v1", "records": len(records),
        "instances": len(instances),
        "by_split": {split: sum(r["split"] == split for r in records) for split in sorted(SPLITS)},
        "missing_or_incomplete": sum(r["window_status"] != "complete" for r in records),
        "paper_protocol": {
            "directions": ["A->B", "B->A"],
            "formation_window": 8,
            "validation_block_length": 16,
            "utility_draws_per_snapshot": 32,
            "equal_source_instance_weight": True,
            "missing_responses_stay_in_coverage_denominator": True,
        },
    }
    (root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
