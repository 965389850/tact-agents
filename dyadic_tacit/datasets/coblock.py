"""CoBlock construction and communication adapter."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from .common import build_record, expand_opportunities, first, instance_id, load_mapping, load_records, load_split_manifest, resolve_split, write_processed

DATASET = "coblock"
DEFAULT_FAMILY = "dependency_construction"
FIELDS = {
    "subgoal": ("subgoal", "own_subgoal", "current_subgoal"),
    "inventory": ("inventory", "own_inventory"),
    "public_construction_state": ("public_construction_state", "construction_state", "public_state"),
    "observable_dependencies": ("observable_dependencies", "dependencies"),
    "executed_partner_actions": ("executed_partner_actions", "partner_actions"),
    "received_messages": ("received_messages", "received_message", "messages"),
}


def collect(input_path, output_path, provider, *, allow_missing=False):
    """Collect provider responses for a CoBlock public-snapshot stream."""
    from .collect import collect_snapshots
    return collect_snapshots(
        DATASET, input_path, output_path, provider,
        allow_missing=allow_missing,
    )


def _context(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row.get("context") if isinstance(row.get("context"), Mapping) else row
    return {name: first(source, *aliases) for name, aliases in FIELDS.items() if first(source, *aliases) is not None}


def _response(row: Mapping[str, Any]) -> dict[str, Any] | None:
    value = row.get("response", row.get("action", row.get("command")))
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "action": value,
        "block_or_region_target": first(row, "block_or_region_target", "block_target", "region_target", "target"),
        "timing": first(row, "response_timing", "timing"),
    }


def process(input_path: str, output_path: str, split_manifest: str | None = None, usage_map: str | None = None) -> dict[str, Any]:
    manifest, mapping, records = load_split_manifest(split_manifest), load_mapping(usage_map), []
    for row in expand_opportunities(load_records(input_path)):
        iid = instance_id(row)
        split = resolve_split(row, manifest, iid)
        response = _response(row)
        valid = first(row, "valid_responses", "legal_responses", "legal_commands", "legal_actions", default=[])
        if isinstance(valid, str):
            valid = [valid]
        warnings = [] if valid else ["valid_response_set_not_supplied"]
        # The native interface permits one command per agent per round.  Preserve
        # source evidence and flag a violation instead of silently dropping rows.
        commands = first(row, "commands_this_round", "round_commands")
        if isinstance(commands, list):
            counts: dict[str, int] = {}
            for command in commands:
                if isinstance(command, Mapping):
                    agent = str(first(command, "agent", "player", default="unknown"))
                    counts[agent] = counts.get(agent, 0) + 1
            if any(count > 1 for count in counts.values()):
                warnings.append("more_than_one_command_per_agent_in_round")
        records.append(build_record(
            dataset=DATASET, row=row,
            family_id=str(first(row, "family_id", "task_family", default=DEFAULT_FAMILY)), split=split,
            context=_context(row), valid_responses=valid, response=response,
            usage_mapping=mapping, warnings=warnings,
        ))
    return write_processed(records, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process observable CoBlock opportunities")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--usage-map")
    args = parser.parse_args()
    print(process(args.input, args.output, args.split_manifest, args.usage_map))


if __name__ == "__main__":
    main()
