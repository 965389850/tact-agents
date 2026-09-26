"""Single masked NLL objective; IDs are metadata, not neural features."""
from collections import defaultdict
from hashlib import sha256
import torch
from torch import nn
from torch.nn import functional as F


def parameter_hash(model):
    h = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(str((name, str(t.dtype), tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def tower(dim, hidden, metric):
    layers = []
    for width in hidden:
        layers.extend([nn.Linear(dim, width), nn.ReLU()])
        dim = width
    return nn.Sequential(*layers, nn.Linear(dim, metric))


class ConditionalResponseModel(nn.Module):
    def __init__(self, context_dim, response_features, catalog, hidden=(128, 128),
                 metric_dim=64, temperature=.2):
        super().__init__()
        if temperature <= 0 or len(catalog) != len(set(catalog)) or not catalog:
            raise ValueError('INVALID_MODEL_SCHEMA')
        self.catalog = tuple(catalog)
        self.temperature = temperature
        self.context_dim = context_dim
        features = torch.tensor(response_features, dtype=torch.float32)
        if features.ndim != 2 or features.shape[0] != len(catalog) or not torch.isfinite(features).all():
            raise ValueError('INVALID_RESPONSE_FEATURES')
        self.register_buffer('response_features', features)
        self.context_tower = tower(context_dim, hidden, metric_dim)
        self.response_tower = tower(features.shape[1], hidden, metric_dim)

    def log_probs(self, context):
        context.validate()
        if len(context.features) != self.context_dim or not set(context.valid_responses) <= set(self.catalog):
            raise ValueError('CONTEXT_SCHEMA_OR_MASK_MISMATCH')
        x = torch.tensor(context.features, dtype=torch.float32, device=self.response_features.device)
        a = F.normalize(self.context_tower(x), dim=-1)
        b = F.normalize(self.response_tower(self.response_features), dim=-1)
        logits = b @ a / self.temperature
        if not torch.isfinite(logits).all():
            raise ValueError('NONFINITE_LOGIT')
        mask = torch.tensor([r in context.valid_responses for r in self.catalog], device=x.device)
        return F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)

    @torch.no_grad()
    def distribution(self, context):
        p = self.log_probs(context).exp().cpu().tolist()
        # Renormalize float32 rounding only; loss uses unrounded log_softmax.
        full = {r: p[i] for i, r in enumerate(self.catalog) if r in context.valid_responses}
        total = sum(full.values())
        return {r: v / total for r, v in full.items()}


class InstanceBalancedNLL:
    def __call__(self, model, records):
        units = defaultdict(lambda: defaultdict(list))
        for context, response in records:
            if response is None:
                continue
            if response not in context.valid_responses:
                raise ValueError('OUTCOME_OUTSIDE_LEGAL_MASK')
            units[context.instance_id][context.key].append(
                -model.log_probs(context)[model.catalog.index(response)])
        if not units:
            raise ValueError('NO_COMPLETE_TRAINING_RECORDS')
        return torch.stack([torch.stack([torch.stack(v).mean() for v in keys.values()]).mean()
                            for keys in units.values()]).mean()


class BoundedReplay:
    """Whole-instance reservoir, label-independent selection, no prompt API."""
    def __init__(self, capacity, records_per_key, rng):
        self.capacity, self.records_per_key, self.rng = capacity, records_per_key, rng
        self.instances = {}
        self.priorities = {}

    def select_records(self, records):
        groups = defaultdict(list)
        for rec in records:
            groups[rec[0].key].append(rec)
        return [rec for group in groups.values()
                for rec in self.rng.sample(group, min(len(group), self.records_per_key))]

    def sample(self, maximum):
        keys = self.rng.sample(sorted(self.instances), min(maximum, len(self.instances)))
        return [rec for key in keys for rec in self.instances[key]]

    def add(self, instance_id, records):
        if instance_id in self.instances:
            raise ValueError('DUPLICATE_REPLAY_INSTANCE')
        self.instances[instance_id] = list(records)
        self.priorities[instance_id] = self.rng.random()
        while sum(map(len, self.instances.values())) > self.capacity:
            victim = min(self.instances, key=lambda k: self.priorities[k])
            del self.instances[victim]
            del self.priorities[victim]
