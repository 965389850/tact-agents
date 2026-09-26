"""Collab-Overcooked adapter following the paper's observable interface."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from .common import (
    build_record, direction, expand_opportunities, first, instance_id, load_mapping, load_records,
    load_split_manifest, resolve_split, write_processed,
)

DATASET = "overcooked"
DEFAULT_FAMILY = "level4_two_ingredient"
FIELDS = {
    "role": ("role", "own_role"),
    "processing_stage": ("processing_stage", "stage", "known_processing_stage"),
    "demand": ("demand", "current_demand", "ingredient_demand"),
    "station_state": ("station_state", "observable_station_state"),
    "handoff_state": ("handoff_state", "observable_handoff_state"),
    "held_object": ("held_object", "own_held_object"),
    "received_coordination_info": ("received_coordination_info", "received_message", "received_hint"),
}


def _context(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row.get("context") if isinstance(row.get("context"), Mapping) else row
    return {name: first(source, *aliases) for name, aliases in FIELDS.items() if first(source, *aliases) is not None}


def _response(row: Mapping[str, Any]) -> dict[str, Any] | None:
    value = row.get("response", row.get("action"))
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return {"action": value, "relation": first(row, "relation", "response_type"), "timing": first(row, "response_timing", "timing")}


def process(input_path: str, output_path: str, split_manifest: str | None = None, usage_map: str | None = None) -> dict[str, Any]:
    manifest = load_split_manifest(split_manifest)
    mapping = load_mapping(usage_map)
    records = []
    for row in expand_opportunities(load_records(input_path)):
        iid = instance_id(row)
        split = resolve_split(row, manifest, iid)
        response = _response(row)
        valid = first(row, "valid_responses", "legal_responses", "legal_actions", default=[])
        if isinstance(valid, str):
            valid = [valid]
        warnings = [] if valid else ["valid_response_set_not_supplied"]
        records.append(build_record(
            dataset=DATASET, row=row,
            family_id=str(first(row, "family_id", "task_family", default=DEFAULT_FAMILY)), split=split,
            context=_context(row), valid_responses=valid, response=response,
            usage_mapping=mapping, warnings=warnings,
        ))
    return write_processed(records, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process observable Collab-Overcooked opportunities")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--usage-map")
    args = parser.parse_args()
    print(process(args.input, args.output, args.split_manifest, args.usage_map))


if __name__ == "__main__":
    main()
