"""Serializable identities, legal pre-decision inputs and fixed usage semantics."""
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math


def stable_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False, ensure_ascii=False).encode()).hexdigest()


def distribution(p):
    if not p or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p.values()):
        raise ValueError('INVALID_DISTRIBUTION')
    if abs(sum(p.values()) - 1) > 1e-6:
        raise ValueError('PROBABILITIES_DO_NOT_SUM_TO_ONE')
    return p


@dataclass(frozen=True)
class Direction:
    self_agent: str
    partner_agent: str
    family_id: str

    def __post_init__(self):
        if self.self_agent == self.partner_agent or not all(asdict(self).values()):
            raise ValueError('INVALID_DIRECTION')


@dataclass(frozen=True, order=True)
class EventOrder:
    episode_index: int
    event_seq: int


@dataclass(frozen=True)
class Context:
    instance_id: str
    instance_index: int
    opportunity_id: str
    direction: Direction
    key: str
    cell: str
    features: tuple
    valid_responses: tuple
    observation_order: EventOrder
    decision_order: EventOrder
    visible_to: str
    source: str

    def validate(self):
        if (self.observation_order.episode_index != self.instance_index or
                self.decision_order.episode_index != self.instance_index or
                not self.observation_order < self.decision_order):
            raise ValueError('OBSERVATION_NOT_PRE_DECISION')
        if self.visible_to != self.direction.self_agent or self.source not in (
                'own_task_interface', 'own_observation', 'own_action_result',
                'visible_partner_completed_action', 'received_message', 'public_transition'):
            raise ValueError('PRIVATE_OR_UNSUPPORTED_OBSERVATION')
        if not self.valid_responses or len(set(self.valid_responses)) != len(self.valid_responses):
            raise ValueError('INVALID_LEGAL_SUPPORT')
        if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in self.features):
            raise ValueError('NON_NUMERIC_FEATURE_OR_TASK_ID')


@dataclass(frozen=True)
class Outcome:
    instance_id: str
    opportunity_id: str
    direction: Direction
    order: EventOrder
    response: str | None
    unknown_reason: str | None = None
    onset: EventOrder | None = None
    completion: EventOrder | None = None
    window_status: str = 'complete'

    @property
    def onset_completion_coobserved(self):
        return self.onset is not None and self.onset == self.completion


class UsageSemantics:
    def __init__(self, mapping, templates, justification, actionability_policy='all_declared_content'):
        # keys are predeclared verification cells, never inferred from outcomes.
        self.mapping = json.loads(json.dumps(mapping))
        self.templates = dict(templates)
        self.justification = justification
        if actionability_policy not in ('all_declared_content', 'no_actionable_content'):
            raise ValueError('UNDECLARED_ACTIONABILITY_POLICY')
        self.actionability_policy = actionability_policy
        if not justification or not mapping:
            raise ValueError('MISSING_USAGE_SEMANTICS')
        # The benchmark response alphabet is dataset-specific.  The original
        # fixture used at most four usage labels, but the appendix's complete
        # Overcooked ontology retains relation and timing, so a fixed four-label
        # ceiling would incorrectly reject a valid response projection.
        for table in self.mapping.values():
            usages = set(table.values())
            if not usages or not usages <= set(templates):
                raise ValueError('INVALID_USAGE_BUDGET_OR_TEMPLATE')
            texts = [templates[u] for u in usages]
            if len(set(texts)) != len(texts):
                raise ValueError('DISTINCT_USAGE_IDENTICAL_TEMPLATE')
        self._hash = stable_hash(dict(mapping=self.mapping, templates=self.templates,
                                      justification=justification, actionability_policy=actionability_policy))

    def schema_hash(self):
        if self._hash != stable_hash(dict(mapping=self.mapping, templates=self.templates,
                                         justification=self.justification, actionability_policy=self.actionability_policy)):
            raise ValueError('MUTATED_USAGE_MAPPING')
        return self._hash

    def label(self, context, response):
        self.schema_hash()
        if response not in context.valid_responses:
            raise ValueError('OUTCOME_OUTSIDE_LEGAL_MASK')
        return self.mapping[context.cell][response]

    def valid_usage_ids(self, context):
        return tuple(sorted({self.label(context, r) for r in context.valid_responses}))

    def pushforward(self, context, full):
        distribution(full)
        if set(full) != set(context.valid_responses):
            raise ValueError('USAGE_MAPPING_SUPPORT_MISMATCH')
        result = dict.fromkeys(self.valid_usage_ids(context), 0.)
        for response, probability in full.items():
            result[self.label(context, response)] += probability
        return distribution(result)


@dataclass(frozen=True)
class ModelVersion:
    direction: Direction
    version_id: str
    parameter_sha256: str
    train_cutoff_instance: int
    train_task_manifest_sha256: str
    context_schema_sha256: str
    usage_map_sha256: str
    response_schema_sha256: str


@dataclass(frozen=True)
class ForecastReceipt:
    prediction_id: str
    context: Context
    model_version: ModelVersion
    model_role: str
    full_probabilities: dict
    usage_probabilities: dict
    selected_probe: bool
    local_deployment_regime_id: str


def validate_carrier(contract):
    missing = []
    for name in ('carrier_id', 'family_id', 'directions', 'full_response',
                 'usage_semantics', 'audit', 'feasibility'):
        if not contract.get(name):
            missing.append(name)
    def walk(obj, path=''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, f'{path}.{key}'.strip('.'))
        elif obj is None:
            missing.append(path)
    walk(contract)
    return sorted(set(missing))
