"""Version-bound empirical screening, never a confidence certificate or GTU gate."""
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import os
from .contracts import distribution, stable_hash


def append_event(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf8') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def half_brier(p, label):
    distribution(p)
    if label not in p:
        raise ValueError('OUTCOME_OUTSIDE_DECLARED_SUPPORT')
    return .5 * sum((v - (k == label)) ** 2 for k, v in p.items())


@dataclass(frozen=True)
class ScreenConfig:
    min_train: int
    min_observed: int
    min_coverage: float
    max_brier: float
    uniform_excess: float = 0.

    def __post_init__(self):
        if (self.min_train < 1 or self.min_observed < 1 or
                not 0 < self.min_coverage <= 1 or not 0 <= self.max_brier <= 1 or
                not 0 <= self.uniform_excess < float('inf')):
            raise ValueError('INVALID_SCREEN_CONFIG')


class ProspectiveScoreLedger:
    def __init__(self, version, training_ids, mapping_hash, block, role, path):
        self.version = version
        self.training_ids = frozenset(training_ids)
        self.mapping_hash = mapping_hash
        self.block = tuple(block)
        self.role = role
        self.path = str(path)
        self.receipts, self.outcomes, self.selected = {}, {}, set()

    def is_first(self, context):
        return (context.instance_id, context.cell) not in self.selected

    def commit(self, receipt):
        context = receipt.context
        context.validate()
        if receipt.prediction_id in self.receipts:
            if self.receipts[receipt.prediction_id] != receipt:
                raise ValueError('CONFLICTING_PREDICTION_ID')
            return
        if (receipt.model_version != self.version or receipt.model_role != self.role or
                context.direction != self.version.direction or
                context.instance_id in self.training_ids or
                context.instance_index <= self.version.train_cutoff_instance or
                not self.block[0] <= context.instance_index <= self.block[1] or
                self.mapping_hash != self.version.usage_map_sha256):
            raise ValueError('PROSPECTIVE_VERSION_OR_PARTITION_MISMATCH')
        distribution(receipt.full_probabilities)
        distribution(receipt.usage_probabilities)
        if receipt.selected_probe != self.is_first(context):
            raise ValueError('PROBE_NOT_FIRST_ACCEPTED_OPPORTUNITY')
        append_event(self.path, {'event': 'forecast', 'receipt': asdict(receipt)})
        self.receipts[receipt.prediction_id] = copy.deepcopy(receipt)
        self.selected.add((context.instance_id, context.cell))

    def observe(self, prediction_id, outcome, semantics):
        r = self.receipts[prediction_id]
        c = r.context
        if prediction_id in self.outcomes:
            if self.outcomes[prediction_id] != outcome:
                raise ValueError('CONFLICTING_OUTCOME')
            return
        if (outcome.instance_id != c.instance_id or outcome.opportunity_id != c.opportunity_id or
                outcome.direction != c.direction or outcome.order.episode_index != c.instance_index or
                not c.decision_order < outcome.order or semantics.schema_hash() != self.mapping_hash):
            raise ValueError('OUTCOME_IDENTITY_OR_CAUSAL_BOUNDARY')
        if outcome.response is None and not outcome.unknown_reason:
            raise ValueError('MISSING_UNKNOWN_REASON')
        if outcome.response is not None:
            semantics.label(c, outcome.response)
        for order in (outcome.onset, outcome.completion):
            if order is not None and not c.decision_order < order <= outcome.order:
                raise ValueError('INVALID_OBSERVABLE_RESPONSE_TIME')
        append_event(self.path, {'event': 'outcome', 'prediction_id': prediction_id,
                                'outcome': asdict(outcome)})
        self.outcomes[prediction_id] = copy.deepcopy(outcome)

    def screen(self, cell, training_count, config, semantics, min_index=0):
        selected = [r for r in self.receipts.values() if r.selected_probe and
                    r.context.cell == cell and r.context.instance_index >= min_index]
        scores, refs = [], []
        unknown = censored = 0
        for r in selected:
            o = self.outcomes.get(r.prediction_id)
            if o is None or o.window_status != 'complete':
                censored += 1
                continue
            if o.response is None:
                unknown += 1
                continue
            u = semantics.label(r.context, o.response)
            scores.append(half_brier(r.usage_probabilities, u))
            refs.append(half_brier(dict.fromkeys(r.usage_probabilities, 1 / len(r.usage_probabilities)), u))
        n, m = len(selected), len(scores)
        coverage = m / n if n else 0.
        loss = sum(scores) / m if m else None
        ref = sum(refs) / m if m else None
        reason = ('INSUFFICIENT_TRAINING_SUPPORT' if training_count < config.min_train else
                  'INSUFFICIENT_FUTURE_SUPPORT' if m < config.min_observed else
                  'LOW_OBSERVATION_COVERAGE' if coverage < config.min_coverage else
                  'EMPIRICAL_SCORE_TOO_HIGH' if loss > config.max_brier else
                  'WORSE_THAN_UNIFORM_REFERENCE' if loss - ref > config.uniform_excess + 1e-12 else
                  'QUALIFIED_EMPIRICAL_ONLY')
        return dict(qualified=reason == 'QUALIFIED_EMPIRICAL_ONLY', reason=reason,
                    selected=n, observed=m, unknown=unknown, censored=censored,
                    coverage=coverage, mean_half_brier=loss, uniform_reference=ref,
                    train_support=training_count, version=asdict(self.version),
                    threshold_hash=stable_hash(asdict(config)), block=self.block)
