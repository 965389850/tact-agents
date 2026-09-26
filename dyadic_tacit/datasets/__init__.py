"""Paper-aligned dataset processing adapters.

Imports are lazy so ``python -m dyadic_tacit.datasets.overcooked`` remains
warning-free.
"""


def process_overcooked(*args, **kwargs):
    from .overcooked import process
    return process(*args, **kwargs)


def collect_hanabi(*args, **kwargs):
    from .collect import collect_snapshots
    return collect_snapshots("hanabi", *args, **kwargs)


def collect_coblock(*args, **kwargs):
    from .collect import collect_snapshots
    return collect_snapshots("coblock", *args, **kwargs)


def train_processed_dataset(*args, **kwargs):
    from .learner import train_processed_dataset as train
    return train(*args, **kwargs)


def process_hanabi(*args, **kwargs):
    from .hanabi import process
    return process(*args, **kwargs)


def process_coblock(*args, **kwargs):
    from .coblock import process
    return process(*args, **kwargs)


__all__ = [
    "collect_coblock", "collect_hanabi",
    "process_coblock", "process_hanabi", "process_overcooked",
    "train_processed_dataset",
]
