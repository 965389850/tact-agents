"""Two-player Hanabi adapter with an explicit hidden-information firewall."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from .common import build_record, expand_opportunities, first, instance_id, load_mapping, load_records, load_split_manifest, resolve_split, write_processed

DATASET = "hanabi"
DEFAULT_FAMILY = "two_player_standard"
FIELDS = {
    "info_tokens": ("info_tokens", "information_tokens"),
    "life_tokens": ("life_tokens", "lives", "life_count"),
    "public_pile_progress": ("public_pile_progress", "pile_progress", "played_cards"),
    "discards": ("discards", "discard_pile", "public_discards"),
    "remaining_cards": ("remaining_cards", "deck_remaining"),
    "visible_partner_cards": ("visible_partner_cards", "partner_cards"),
    "own_card_knowledge": ("own_card_knowledge", "received_hints", "card_knowledge"),
}


def collect(input_path, output_path, provider, *, allow_missing=False):
    """Collect provider responses for a Hanabi public-snapshot stream."""
    from .collect import collect_snapshots
    return collect_snapshots(
        DATASET, input_path, output_path, provider,
        allow_missing=allow_missing,
    )
FORBIDDEN = {
    "own_cards", "true_own_cards", "private_hand", "hidden_hand", "future_deck",
    "deck_order", "remaining_deck_order", "partner_private_plan", "private_plan", "chain_of_thought",
}


def _context(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row.get("context") if isinstance(row.get("context"), Mapping) else row
    result = {name: first(source, *aliases) for name, aliases in FIELDS.items() if first(source, *aliases) is not None}
    # Rule metadata is public and fixed by the paper's standard two-player game.
    rules = first(row, "rules", "game_rules")
    if isinstance(rules, Mapping):
        result["rules"] = {
            "players": rules.get("players", 2), "colors": rules.get("colors", 5),
            "ranks": rules.get("ranks", 5), "hand_size": rules.get("hand_size", 5),
            "info_tokens": rules.get("info_tokens", 8), "life_tokens": rules.get("life_tokens", 3),
        }
        expected = {"players": 2, "colors": 5, "ranks": 5, "hand_size": 5, "info_tokens": 8, "life_tokens": 3}
        if result["rules"] != expected:
            raise ValueError(f"Hanabi source rules {result['rules']} do not match the paper's standard two-player rules {expected}")
    else:
        result["rules"] = {"players": 2, "colors": 5, "ranks": 5, "hand_size": 5, "info_tokens": 8, "life_tokens": 3}
    return result


def _response(row: Mapping[str, Any]) -> dict[str, Any] | None:
    value = row.get("response", row.get("action"))
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "action_type": value,
        "card_position": first(row, "card_position", "card_index"),
        "hint_referent": first(row, "hint_referent", "hint_color", "hint_rank"),
        "timing": first(row, "response_timing", "timing"),
    }


def process(input_path: str, output_path: str, split_manifest: str | None = None, usage_map: str | None = None) -> dict[str, Any]:
    manifest, mapping, records = load_split_manifest(split_manifest), load_mapping(usage_map), []
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
            usage_mapping=mapping, warnings=warnings, forbidden_context_keys=FORBIDDEN,
        ))
    return write_processed(records, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process observable two-player Hanabi opportunities")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest")
    parser.add_argument("--usage-map")
    args = parser.parse_args()
    print(process(args.input, args.output, args.split_manifest, args.usage_map))


if __name__ == "__main__":
    main()
