"""Host boundary for independent directional learners."""
from dataclasses import asdict
import copy
from .contracts import stable_hash
from .model import parameter_hash


def render_private_prompt(base_prompt, guidance):
    """Inject only the read-only self-side guidance into the prompt slot."""
    return base_prompt + (
        '\n<private_self_guidance>\n' + guidance + '\n</private_self_guidance>'
        if guidance else ''
    )


class DistributionalSelfProjector:
    def __init__(self, semantics, max_chars, char_counter=None):
        self.semantics, self.max_chars = semantics, max_chars
        # The appendix specifies a character budget, rather than a provider
        # token budget.  Keep the counter injectable for provider integrations.
        self.char_counter = char_counter or len

    def project(self, context, usage, qualified):
        if not qualified:
            return None, 'UNQUALIFIED'
        policy = self.semantics.actionability_policy
        if policy == 'no_actionable_content':
            return None, 'NO_ACTIONABLE_USAGE_CONTENT'
        lines = [f'{u}: {usage[u]:.3f}. {self.semantics.templates[u]}' for u in sorted(usage)]
        text = ('Tentative partner-relative possibilities, not guarantees.\n' + '\n'.join(lines) +
                '\nRecheck your own current legal observations; disregard contradictions.')
        if self.char_counter(text) <= self.max_chars:
            return text, 'DISTRIBUTIONAL_GUIDANCE'

        # Large response alphabets (notably Overcooked) cannot fit all
        # templates in the common 600-character slot.  Keep a deterministic
        # probability-only projection, then add an omitted-mass bucket if
        # necessary; never emit an over-budget prompt or fail the host run.
        ordered = sorted(usage, key=lambda label: (-usage[label], label))
        prefix, suffix = 'Possibilities: ', '\nRecheck legal observations.'
        kept, omitted = [], 0.0
        for label in ordered:
            candidate = prefix + '; '.join(kept + [f'{label}={usage[label]:.3f}']) + suffix
            if self.char_counter(candidate) > self.max_chars:
                omitted += usage[label]
                continue
            kept.append(f'{label}={usage[label]:.3f}')
        if omitted > 0 and kept:
            candidate = prefix + '; '.join(kept + [f'OTHER={omitted:.3f}']) + suffix
            while self.char_counter(candidate) > self.max_chars and len(kept) > 1:
                label = kept.pop()
                value = float(label.rsplit('=', 1)[-1])
                omitted += value
                candidate = prefix + '; '.join(kept + [f'OTHER={omitted:.3f}']) + suffix
        text = candidate if kept else (prefix + f'OTHER={sum(usage.values()):.3f}' + suffix)
        if self.char_counter(text) > self.max_chars:
            text = text[:self.max_chars]
        return text, 'DISTRIBUTIONAL_GUIDANCE_COMPACT'


class ReadOnlyActiveView:
    def __init__(self, direction, semantics, active, max_chars):
        self.direction = direction
        self.semantics = copy.deepcopy(semantics)
        # Deliberately exclude optimizer, train, candidate, replay and monitoring ledger.
        self._model = copy.deepcopy(active['model']) if active else None
        self.version = copy.deepcopy(active['version']) if active else None
        self._qualified = copy.deepcopy(active['qualified']) if active else {}
        self._disabled = copy.deepcopy(active['disabled']) if active else {}
        self.projector = DistributionalSelfProjector(self.semantics, max_chars)

    def state_hash(self):
        return stable_hash(dict(direction=asdict(self.direction), version=asdict(self.version) if self.version else None,
            parameters=parameter_hash(self._model) if self._model else None,
            qualified=self._qualified, disabled=self._disabled, usage=self.semantics.schema_hash()))

    def readout(self, context):
        context.validate()
        if context.direction != self.direction:
            raise ValueError('PARTNER_STATE_READ_FORBIDDEN')
        if self._model is None:
            return dict(guidance=None, reason='NO_ACTIVE_MODEL', full_p=None, usage_p=None)
        if parameter_hash(self._model) != self.version.parameter_sha256:
            raise ValueError('READ_ONLY_MODEL_MUTATED')
        full = self._model.distribution(context)
        usage = self.semantics.pushforward(context, full)
        guidance, reason = self.projector.project(context, usage,
            context.cell in self._qualified and context.cell not in self._disabled)
        return dict(guidance=guidance, reason=reason, full_p=full, usage_p=usage,
                    version=asdict(self.version))


class IndependentDyadRunnerV3:
    """Host emits one focal decision at a time and progresses all intervening steps.

    Host supplies legal_contexts/task reset, execute(context, private_prompt) -> Outcome.
    It never receives learners or an evaluator joint-state summary.
    The host supplies this protocol without exposing learner state to the
    provider.
    """
    def __init__(self, learners):
        self.learners = dict(learners)
        if len(learners) != 2 or len({id(x) for x in learners.values()}) != 2:
            raise ValueError('TWO_INDEPENDENT_LEARNERS_REQUIRED')
        a, b = learners.values()
        if (a.direction.self_agent != b.direction.partner_agent or
                a.direction.partner_agent != b.direction.self_agent or
                a.direction.family_id != b.direction.family_id or a.optimizer is b.optimizer):
            raise ValueError('DIRECTION_PAIR_MISMATCH')
        if {p.data_ptr() for p in a.train.parameters()} & {p.data_ptr() for p in b.train.parameters()}:
            raise ValueError('SHARED_WRITABLE_PARAMETERS')

    def run(self, host, instance_id, index):
        host.reset(instance_id, index)
        views = {}
        for agent, learner in self.learners.items():
            learner.begin_instance(instance_id, index)
            views[agent] = learner.freeze_for_audit()
        calls, exposures = 0, 0
        for context in host.decisions():
            learner = self.learners[context.direction.self_agent]
            learner.forecast(context)
            result = views[context.direction.self_agent].readout(context)
            base = host.base_prompt(context)
            guidance = result['guidance']
            prompt = render_private_prompt(base, guidance)
            outcome = host.execute(context, prompt)
            learner.observe(outcome)
            calls += 1
            exposures += int(guidance is not None)
            host.log_prompt(dict(agent=context.direction.self_agent, instance_id=instance_id,
                base_hash=stable_hash(base), guidance_hash=stable_hash(guidance),
                final_hash=stable_hash(prompt), guidance_present=guidance is not None,
                guidance_bytes=len((guidance or '').encode()), readout_reason=result['reason']))
        return dict(updates={a: l.finish_instance() for a, l in self.learners.items()},
                    decisions=calls, guidance_exposures=exposures)


def audit_window(host, views, instance_id, index):
    before = {a: v.state_hash() for a, v in views.items()}
    host.reset(instance_id, index)
    outcomes = []
    for context in host.decisions():
        result = views[context.direction.self_agent].readout(context)
        prompt = render_private_prompt(host.base_prompt(context), result['guidance'])
        outcomes.append(asdict(host.execute(context, prompt)))
    if before != {a: v.state_hash() for a, v in views.items()}:
        raise ValueError('AUDIT_STATE_MUTATED')
    return dict(full_C_outcomes=outcomes, learner_updates=0, before_hashes=before,
                after_hashes={a: v.state_hash() for a, v in views.items()},
                status='FROZEN_AUDIT_COMPLETE')
