"""Train each instance; verify immutable candidates; deploy only at boundaries."""
from collections import defaultdict
from dataclasses import asdict
import copy
import os
from pathlib import Path
import random
import tempfile
import torch
from .contracts import ForecastReceipt, ModelVersion, stable_hash
from .evidence import ProspectiveScoreLedger, ScreenConfig, append_event
from .model import BoundedReplay, InstanceBalancedNLL, parameter_hash


def frozen(model):
    result = copy.deepcopy(model).eval()
    for p in result.parameters():
        p.requires_grad_(False)
    return result


class DirectionalLearnerV3:
    def __init__(self, direction, model, semantics, cfg, output, seed):
        self.direction, self.train, self.semantics = direction, model, semantics
        self.cfg, self.output = copy.deepcopy(cfg), str(output)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                           weight_decay=cfg['weight_decay'])
        self.rng = random.Random(seed)
        self.replay = BoundedReplay(cfg['replay_capacity'], cfg['records_per_key'], self.rng)
        self.support = defaultdict(set)
        self.seen, self.completed = set(), 0
        self.candidate = self.active = None
        self.current = None
        self.records, self.pending, self.events = [], {}, []
        self.identity = stable_hash(dict(direction=asdict(direction), config=cfg,
                                         usage=semantics.schema_hash(), catalog=model.catalog,
                                         initialization_hash=parameter_hash(model), seed=seed))

    def _snapshot(self, cutoff):
        model = frozen(self.train)
        ph = parameter_hash(model)
        version = ModelVersion(self.direction, f'{cutoff}:{ph[:16]}', ph, cutoff,
                               stable_hash(sorted(self.seen)), self.cfg['context_schema_hash'],
                               self.semantics.schema_hash(), stable_hash(model.catalog))
        start, end = cutoff + 1, cutoff + self.cfg['block']
        return dict(model=model, version=version, training_ids=frozenset(self.seen),
                    support=copy.deepcopy(dict(self.support)), qualified={}, disabled={},
                    ledger=ProspectiveScoreLedger(version, self.seen, self.semantics.schema_hash(),
                        (start, end), 'candidate', Path(self.output) / 'predictions.jsonl'))

    def begin_instance(self, instance_id, index, partition='formation'):
        if partition != 'formation':
            raise ValueError('AUDIT_OR_UTILITY_CANNOT_ENTER_LEARNING')
        if self.current is not None or instance_id in self.seen or index != self.completed + 1:
            raise ValueError('INSTANCE_ORDER_OR_DUPLICATE')
        self.current = (instance_id, index)
        self.records, self.pending = [], {}
        self._pinned = self.active_identity()

    def active_identity(self):
        if self.active is None:
            return None
        return stable_hash(dict(version=asdict(self.active['version']),
                                qualified=self.active['qualified'], disabled=self.active['disabled']))

    def forecast(self, context):
        context.validate()
        if self.current != (context.instance_id, context.instance_index) or context.direction != self.direction:
            raise ValueError('WRONG_INSTANCE_OR_DIRECTION')
        if self.active_identity() != self._pinned or context.opportunity_id in self.pending:
            raise ValueError('INTRA_INSTANCE_STATE_CHANGE_OR_DUPLICATE_OPPORTUNITY')
        receipts = []
        for role, slot in (('candidate', self.candidate), ('active', self.active)):
            if slot is None:
                continue
            if parameter_hash(slot['model']) != slot['version'].parameter_sha256:
                raise ValueError('IMMUTABLE_MODEL_MUTATED')
            p = slot['model'].distribution(context)
            u = self.semantics.pushforward(context, p)
            ledger = slot['ledger'] if role == 'candidate' else slot['monitor']
            receipt = ForecastReceipt(stable_hash([self.identity, context.instance_id,
                context.opportunity_id, role, slot['version'].version_id]), context,
                slot['version'], role, p, u, ledger.is_first(context), str(self._pinned))
            ledger.commit(receipt)  # durable before returning to host action selection
            receipts.append(receipt)
        self.pending[context.opportunity_id] = (context, receipts, False)
        return receipts

    def observe(self, outcome):
        context, receipts, done = self.pending[outcome.opportunity_id]
        if done:
            raise ValueError('DUPLICATE_OPPORTUNITY_OUTCOME')
        if (outcome.direction != self.direction or outcome.instance_id != context.instance_id or
                outcome.order.episode_index != context.instance_index or
                not context.decision_order < outcome.order):
            raise ValueError('INVALID_OUTCOME_CAUSAL_IDENTITY')
        if outcome.response is None and not outcome.unknown_reason:
            raise ValueError('MISSING_UNKNOWN_REASON')
        for r in receipts:
            slot = self.candidate if r.model_role == 'candidate' else self.active
            ledger = slot['ledger'] if r.model_role == 'candidate' else slot['monitor']
            ledger.observe(r.prediction_id, outcome, self.semantics)
        if outcome.response is not None:
            self.semantics.label(context, outcome.response)
        # Preserve every opportunity, including UNKNOWN, before training selection.
        append_event(Path(self.output) / 'outcomes.jsonl',
                     dict(context=asdict(context), outcome=asdict(outcome)))
        response = outcome.response if outcome.window_status == 'complete' else None
        self.records.append((context, response))
        self.pending[context.opportunity_id] = (context, receipts, True)

    def _monitor(self, index):
        if not self.active:
            return
        a = self.active
        for cell in a['qualified']:
            if cell in a['disabled']:
                continue
            ledger = a['monitor']
            valid_indices = [r.context.instance_index for pid, r in ledger.receipts.items()
                if r.selected_probe and r.context.cell == cell and pid in ledger.outcomes
                and ledger.outcomes[pid].response is not None
                and ledger.outcomes[pid].window_status == 'complete']
            last = max(valid_indices, default=a['deployed_at'])
            result = ledger.screen(cell, len(a['support'].get(cell, ())),
                ScreenConfig(**self.cfg['screens'][cell]), self.semantics,
                min_index=index - self.cfg['monitor_window'] + 1)
            if index - last > self.cfg['max_age']:
                a['disabled'][cell] = 'EXPIRED_UNSUPPORTED'
            elif result['observed'] >= self.cfg['monitor_min_observed'] and not result['qualified']:
                a['disabled'][cell] = result['reason']
            append_event(Path(self.output) / 'monitor.jsonl', dict(index=index, cell=cell,
                screen=result, disabled=a['disabled'].get(cell), applies_from=index + 1))

    def finish_instance(self):
        if self.current is None or any(not x[2] for x in self.pending.values()):
            raise ValueError('OPEN_RESPONSE_WINDOWS')
        instance_id, index = self.current
        selected = self.replay.select_records(self.records)
        batch = selected + self.replay.sample(self.cfg['replay_instances'])
        usable = [rec for rec in batch if rec[1] is not None]
        loss_value = None
        if usable:
            loss_value = self.fit_records(usable)
            for context, _ in usable:
                self.support[context.cell].add(context.instance_id)
        self.replay.add(instance_id, selected)
        self.seen.add(instance_id)
        self.completed = index
        self._monitor(index)
        promoted = None
        if index >= self.cfg['warmup'] and (index - self.cfg['warmup']) % self.cfg['block'] == 0:
            if self.candidate is not None:
                c = self.candidate
                if c['ledger'].block[1] != index:
                    raise ValueError('VALIDATION_BLOCK_MISMATCH')
                results = self.candidate_results(c)
                qualified = {cell: r for cell, r in results.items() if r['qualified']}
                append_event(Path(self.output) / 'qualification.jsonl', dict(index=index, cells=results))
                if qualified:
                    self.active = copy.deepcopy(c)
                    self.active.update(qualified=qualified, disabled={}, deployed_at=index + 1)
                    self.active.pop('ledger')
                    self.active['monitor'] = ProspectiveScoreLedger(c['version'], c['training_ids'],
                        self.semantics.schema_hash(), (index + 1, 2**63 - 1), 'active',
                        Path(self.output) / 'predictions.jsonl')
                    promoted = c['version'].version_id
            self.candidate = self._snapshot(index)
        event = dict(completed=index, train_hash=parameter_hash(self.train), loss=loss_value,
            candidate_cutoff=self.candidate['version'].train_cutoff_instance if self.candidate else None,
            active_cutoff=self.active['version'].train_cutoff_instance if self.active else None,
            promoted=promoted, active_identity=self.active_identity())
        self.events.append(event)
        append_event(Path(self.output) / 'deployments.jsonl', event)
        self.current = None
        return event

    def training_loss(self, records):
        return InstanceBalancedNLL()(self.train, records)

    def fit_records(self, records):
        self.train.train()
        for _ in range(self.cfg['steps']):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.training_loss(records)
            if not torch.isfinite(loss):
                raise ValueError('NONFINITE_TRAINING_LOSS')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.train.parameters(), self.cfg['grad_clip'])
            self.optimizer.step()
        return float(loss.detach())

    def candidate_results(self, candidate):
        return {cell: candidate['ledger'].screen(cell, len(candidate['support'].get(cell, ())),
            ScreenConfig(**screen), self.semantics) for cell, screen in self.cfg['screens'].items()}

    def freeze_for_audit(self):
        from .runtime import ReadOnlyActiveView
        return ReadOnlyActiveView(self.direction, self.semantics, self.active,
                                  self.cfg.get('max_guidance_chars',
                                               self.cfg.get('max_guidance_tokens', 600)))

    def save(self, path):
        if self.current is not None:
            raise ValueError('CHECKPOINT_REQUIRES_INSTANCE_BOUNDARY')
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        portable = copy.deepcopy(self)
        # Keep log locations relocatable instead of serializing the caller's
        # local filesystem path.
        portable.output = f'events-{self.direction.self_agent}'
        for slot in (portable.candidate, portable.active):
            if slot is not None:
                ledger = slot.get('ledger', slot.get('monitor'))
                ledger.path = str(Path(portable.output) / 'predictions.jsonl')
        portable.records = []
        portable.pending = {}
        portable.events = []
        payload = dict(schema_version=3, identity=self.identity, learner=portable,
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       checkpoint_commit_marker='COMPLETE')
        fd, temp = tempfile.mkstemp(dir=target.parent, prefix='.pending-')
        try:
            with os.fdopen(fd, 'wb') as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            loaded = torch.load(temp, map_location=self.train.response_features.device, weights_only=False)
            if loaded['identity'] != self.identity or loaded['learner'].state_hash() != self.state_hash():
                raise ValueError('CHECKPOINT_ROUNDTRIP_MISMATCH')
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    @staticmethod
    def load(path, expected_identity, device='cpu', output=None):
        # Checkpoints are trusted project-generated state files.
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get('schema_version') != 3 or payload.get('identity') != expected_identity or payload.get('checkpoint_commit_marker') != 'COMPLETE':
            raise ValueError('LEGACY_OR_INCOMPATIBLE_CHECKPOINT')
        # Restore the learner state without replacing the caller's global RNG.
        learner = payload['learner']
        learner.output = str(Path(output) if output is not None else Path(path).parent / learner.output)
        for slot in (learner.candidate, learner.active):
            if slot is not None:
                ledger = slot.get('ledger', slot.get('monitor'))
                ledger.path = str(Path(learner.output) / 'predictions.jsonl')
        return learner

    def state_hash(self):
        # Includes all behavioral and optimization state, deterministic tensor serialization.
        def normalize(x):
            if isinstance(x, torch.Tensor):
                return [str(x.dtype), list(x.shape), x.detach().cpu().tolist()]
            if isinstance(x, dict):
                return {str(k): normalize(v) for k, v in x.items()}
            if isinstance(x, (tuple, list)):
                return [normalize(v) for v in x]
            return x
        def slot_state(slot):
            if slot is None:
                return None
            ledger = slot.get('ledger', slot.get('monitor'))
            return dict(version=asdict(slot['version']), model=parameter_hash(slot['model']),
                receipts={k: asdict(v) for k, v in ledger.receipts.items()},
                outcomes={k: asdict(v) for k, v in ledger.outcomes.items()},
                selected=sorted(ledger.selected), block=ledger.block,
                support={k: sorted(v) for k, v in slot['support'].items()},
                qualified=slot['qualified'], disabled=slot['disabled'],
                deployed_at=slot.get('deployed_at'))
        return stable_hash(dict(identity=self.identity, completed=self.completed,
            train=parameter_hash(self.train), optimizer=normalize(self.optimizer.state_dict()),
            candidate=slot_state(self.candidate), active=slot_state(self.active),
            seen=sorted(self.seen), rng=normalize(self.rng.getstate()), replay_priorities=self.replay.priorities,
            support={k: sorted(v) for k, v in self.support.items()},
            replay={k: [(asdict(c), r) for c, r in records] for k, records in self.replay.instances.items()}))
