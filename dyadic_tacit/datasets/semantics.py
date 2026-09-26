"""Versioned complete-response and self-side usage mappings.

The appendix separates the observable complete response
``C=(R,G,T_resp)`` from the self-side usage projection ``U``. The response
catalogs below are finite and stable for the learner; raw target fields stay
in processed records for audit and replay.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import UsageSemantics


def _timing(value: Any, default: str) -> str:
    raw = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "NOW": default, "SAME_ROUND": default, "SAME_ACTION": default,
        "SAME_COMMAND": default, "ANTICIPATORY": "ANTICIPATORY",
        "SYNCHRONIZED": "SYNCHRONIZED", "SYNC": "SYNCHRONIZED",
        "DELAYED": "DELAYED", "ABSENT": "ABSENT", "NONE": "ABSENT",
    }
    return aliases.get(raw, default)


# Overcooked complete responses follow the original handoff ontology.
_OV_TARGETS = {
    "HANDOFF": ("ACCEPT_UPSTREAM_OUTPUT", "WAIT_FOR_HANDOFF"),
    "COMPLEMENT": ("PREPARE_DOWNSTREAM_HANDOFF", "EXECUTE_COMPLEMENTARY_STAGE"),
    "PARALLEL": ("EXECUTE_COMPLEMENTARY_STAGE",),
    "DEFER": ("WAIT_FOR_HANDOFF", "MAINTAIN_CURRENT_STAGE"),
    "CONFLICT": ("MAINTAIN_CURRENT_STAGE", "RELEASE_SHARED_RESOURCE"),
    "NONE": ("NO_COORDINATION_TARGET",),
}
_OV_TIMINGS = ("ANTICIPATORY", "SYNCHRONIZED", "DELAYED", "ABSENT")
OVERCOOKED_RESPONSES = tuple(
    f"{relation}|{target}|{timing}"
    for relation, targets in _OV_TARGETS.items()
    for target in targets
    for timing in (("ABSENT",) if relation == "NONE" else _OV_TIMINGS)
)
OVERCOOKED_CELLS = ("OVERCOOKED|HANDOFF|NOW", "OVERCOOKED|COMPLEMENT|NOW")


def _overcooked_usage(response: str) -> str:
    relation, _target, timing = response.split("|", 2)
    return f"OV_{relation}_{timing}"


OVERCOOKED_USAGE = {
    cell: {response: _overcooked_usage(response) for response in OVERCOOKED_RESPONSES}
    for cell in OVERCOOKED_CELLS
}
OVERCOOKED_TEMPLATES = {
    usage: (
        f"Partner response relation {usage[3:].rsplit('_', 1)[0]} with timing "
        f"{usage.rsplit('_', 1)[-1]}; use only the observable handoff state."
    )
    for usage in sorted({_overcooked_usage(response) for response in OVERCOOKED_RESPONSES})
}


def _canonical_overcooked(response: Any, row: Mapping[str, Any]) -> str:
    value = response if isinstance(response, Mapping) else {"action": response}
    relation = str(value.get("relation", value.get("response_type", ""))).strip().upper()
    target = str(value.get("target", value.get("response_target", ""))).strip().upper()
    target = {
        "ACCEPT": "ACCEPT_UPSTREAM_OUTPUT", "HANDOFF": "ACCEPT_UPSTREAM_OUTPUT",
        "PREPARE": "PREPARE_DOWNSTREAM_HANDOFF", "COMPLEMENT": "EXECUTE_COMPLEMENTARY_STAGE",
        "PARALLEL": "EXECUTE_COMPLEMENTARY_STAGE", "WAIT": "WAIT_FOR_HANDOFF",
        "DEFER": "WAIT_FOR_HANDOFF", "CONFLICT": "MAINTAIN_CURRENT_STAGE",
        "NONE": "NO_COORDINATION_TARGET",
    }.get(target, target)
    timing = _timing(value.get("timing", value.get("response_timing")), "SYNCHRONIZED")
    candidate = (
        "NONE|NO_COORDINATION_TARGET|ABSENT" if relation == "NONE"
        else f"{relation}|{target}|{timing}"
    )
    if candidate in OVERCOOKED_RESPONSES:
        return candidate
    context = row.get("context") if isinstance(row.get("context"), Mapping) else row
    if "handoff_state" in context or "observable_handoff_state" in context:
        return "HANDOFF|ACCEPT_UPSTREAM_OUTPUT|SYNCHRONIZED"
    return "COMPLEMENT|EXECUTE_COMPLEMENTARY_STAGE|SYNCHRONIZED"


# Hanabi keeps response family, target class, and timing in complete C. Actual
# card positions and hint referents remain in the raw response object.
HANABI_RESPONSES = (
    "PLAY|CARD_POSITION|SAME_ACTION",
    "DISCARD|CARD_POSITION|SAME_ACTION",
    "HINT_COLOR|HINT_REFERENT|SAME_ACTION",
    "HINT_RANK|HINT_REFERENT|SAME_ACTION",
)
HANABI_USAGE = {
    "HANABI|PARTNER|NOW": {
        "PLAY|CARD_POSITION|SAME_ACTION": "PARTNER_PLAY",
        "DISCARD|CARD_POSITION|SAME_ACTION": "PARTNER_DISCARD",
        "HINT_COLOR|HINT_REFERENT|SAME_ACTION": "PARTNER_HINT",
        "HINT_RANK|HINT_REFERENT|SAME_ACTION": "PARTNER_HINT",
    }
}
HANABI_TEMPLATES = {
    "PARTNER_PLAY": "Partner play may change public fireworks; reassess the legal self action.",
    "PARTNER_DISCARD": "Partner discard may change the discard pile and information tokens; reassess timing.",
    "PARTNER_HINT": "Partner hint provides public information; use only the observed hint referent.",
}


def _canonical_hanabi(response: Any, row: Mapping[str, Any]) -> str | None:
    value = response if isinstance(response, Mapping) else {"action_type": response}
    action = str(value.get("action_type", value.get("action", value.get("type", "")))).lower()
    action = {"reveal_color": "hint_color", "reveal_rank": "hint_rank"}.get(action, action)
    if action == "play":
        return "PLAY|CARD_POSITION|SAME_ACTION"
    if action == "discard":
        return "DISCARD|CARD_POSITION|SAME_ACTION"
    if action in {"hint_color", "hint_rank"}:
        return f"{action.upper()}|HINT_REFERENT|SAME_ACTION"
    return None


# CoBlock keeps construction/communication family, target class, and command
# timing in C. Concrete block/region IDs remain in raw records.
COBLOCK_RESPONSES = (
    "PLACE|BLOCK_OR_REGION|SAME_COMMAND",
    "BREAK|BLOCK_OR_REGION|SAME_COMMAND",
    "MESSAGE|PUBLIC_COORDINATION|SAME_COMMAND",
    "WAIT|DEPENDENCY|SAME_COMMAND",
)
COBLOCK_USAGE = {
    "COBLOCK|PARTNER|NOW": {
        "PLACE|BLOCK_OR_REGION|SAME_COMMAND": "PARTNER_CONSTRUCTS",
        "BREAK|BLOCK_OR_REGION|SAME_COMMAND": "PARTNER_RELIEVES_BLOCKAGE",
        "MESSAGE|PUBLIC_COORDINATION|SAME_COMMAND": "PARTNER_COMMUNICATES",
        "WAIT|DEPENDENCY|SAME_COMMAND": "PARTNER_DEFERS",
    }
}
COBLOCK_TEMPLATES = {
    "PARTNER_CONSTRUCTS": "Partner construction may advance a shared dependency; reassess construction timing.",
    "PARTNER_RELIEVES_BLOCKAGE": "Partner blockage relief may change the public dependency; reassess the next legal step.",
    "PARTNER_COMMUNICATES": "Partner communication changes public coordination information; use only the received message.",
    "PARTNER_DEFERS": "Partner deferral leaves the dependency open; reassess whether to wait or continue.",
}


def _canonical_coblock(response: Any, row: Mapping[str, Any]) -> str | None:
    value = response if isinstance(response, Mapping) else {"action": response}
    action = str(value.get("action", value.get("action_type", value.get("type", "")))).lower()
    action = {"construct": "place", "remove": "break", "communicate": "message"}.get(action, action)
    if action == "place":
        return "PLACE|BLOCK_OR_REGION|SAME_COMMAND"
    if action == "break":
        return "BREAK|BLOCK_OR_REGION|SAME_COMMAND"
    if action == "message":
        return "MESSAGE|PUBLIC_COORDINATION|SAME_COMMAND"
    if action == "wait":
        return "WAIT|DEPENDENCY|SAME_COMMAND"
    return None


DATASET_RESPONSE_CATALOGS = {
    "overcooked": OVERCOOKED_RESPONSES,
    "hanabi": HANABI_RESPONSES,
    "coblock": COBLOCK_RESPONSES,
}


def canonical_response(dataset: str, response: Any, row: Mapping[str, Any]) -> str | None:
    if dataset == "overcooked":
        return _canonical_overcooked(response, row)
    if dataset == "hanabi":
        return _canonical_hanabi(response, row)
    if dataset == "coblock":
        return _canonical_coblock(response, row)
    raise ValueError(f"UNKNOWN_DATASET_RESPONSE_SCHEMA:{dataset}")


def response_catalog(dataset: str) -> tuple[str, ...]:
    try:
        return DATASET_RESPONSE_CATALOGS[dataset]
    except KeyError as exc:
        raise ValueError(f"UNKNOWN_DATASET_RESPONSE_SCHEMA:{dataset}") from exc


def response_cell(dataset: str, row: Mapping[str, Any]) -> str:
    if dataset == "overcooked":
        context = row.get("context") if isinstance(row.get("context"), Mapping) else row
        return (
            OVERCOOKED_CELLS[0]
            if context.get("handoff_state", context.get("observable_handoff_state")) is not None
            else OVERCOOKED_CELLS[1]
        )
    if dataset == "hanabi":
        return "HANABI|PARTNER|NOW"
    if dataset == "coblock":
        return "COBLOCK|PARTNER|NOW"
    raise ValueError(f"UNKNOWN_DATASET_RESPONSE_SCHEMA:{dataset}")


def semantic_usage(dataset: str, response: Any, row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return the canonical complete-response label and its usage label."""
    canonical = canonical_response(dataset, response, row)
    if canonical is None:
        return None, None
    semantics = semantics_for(dataset)
    usage = semantics.mapping.get(response_cell(dataset, row), {}).get(canonical)
    return canonical, usage


def semantics_for(dataset: str) -> UsageSemantics:
    if dataset == "overcooked":
        return UsageSemantics(
            OVERCOOKED_USAGE, OVERCOOKED_TEMPLATES,
            "Complete Overcooked response keeps relation, target, and response timing; usage projects to relation and timing.",
            actionability_policy="all_declared_content",
        )
    if dataset == "hanabi":
        return UsageSemantics(
            HANABI_USAGE, HANABI_TEMPLATES,
            "Complete Hanabi response keeps play/discard/hint family, target class, and timing; color and rank hints share the information usage.",
            actionability_policy="all_declared_content",
        )
    if dataset == "coblock":
        return UsageSemantics(
            COBLOCK_USAGE, COBLOCK_TEMPLATES,
            "Complete CoBlock response keeps construction or communication family, target class, and command timing; concrete block IDs remain auditable fields.",
            actionability_policy="all_declared_content",
        )
    raise ValueError(f"UNKNOWN_DATASET_SEMANTICS:{dataset}")


def semantics_metadata(dataset: str) -> dict[str, Any]:
    semantics = semantics_for(dataset)
    return {
        "dataset": dataset,
        "schema_hash": semantics.schema_hash(),
        "response_catalog": list(response_catalog(dataset)),
        "mapping": semantics.mapping,
        "templates": semantics.templates,
        "justification": semantics.justification,
        "actionability_policy": semantics.actionability_policy,
        "complete_response_schema": "C=(R,G,T_resp)",
        "usage_schema": "U=self_side_participation_projection(C)",
    }
