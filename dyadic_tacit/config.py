"""Production defaults for the directional learner."""

from .contracts import stable_hash


def default_config() -> dict:
    """Return the paper-aligned learner defaults used by dataset adapters."""
    return dict(
        learning_rate=1e-3,
        weight_decay=1e-4,
        steps=4,
        grad_clip=1.0,
        replay_capacity=512,
        records_per_key=4,
        replay_instances=8,
        warmup=8,
        block=16,
        screens={
            "default_cell": dict(
                min_train=3,
                min_observed=3,
                min_coverage=0.8,
                max_brier=0.30,
                uniform_excess=0.0,
            )
        },
        monitor_window=8,
        monitor_min_observed=3,
        max_age=8,
        context_schema_hash=stable_hash({"features": "dataset_specific", "scope": "paper_method"}),
        max_guidance_chars=600,
    )
