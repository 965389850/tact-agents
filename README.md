# TACT

Official method implementation for **TACT** (*Tacit Agent Coordination via Interaction-specific Training*), from:

> **Do Agents Develop Tacit Understanding? From Repeated Collaboration to Partner-Specific Coordination**

This release contains the paper method core and dataset adapters for converting observable interaction records into directional learner records. It intentionally excludes benchmark environments, baselines, experiment dispatch, and scientific audit scripts.

## Supported providers

The provider layer contains the three backbones used in the paper:

- GPT through the OpenAI Chat Completions API;
- Claude through the Anthropic Messages API;
- Gemini through the Google Generative Language API.

Set the provider-specific API key before creating a provider:

```bash
export OPENAI_API_KEY="..."
# or ANTHROPIC_API_KEY / GEMINI_API_KEY
```

Select a provider with `DTU_LLM_PROVIDER` (`gpt`, `claude`, or `gemini`) and override its model with `DTU_LLM_MODEL`.

## Installation

```bash
cd submit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.10 or newer is required.

## Package layout

```text
submit/
├── dyadic_tacit/
│   ├── datasets/       # Dataset processors, collectors, and learner adapter
│   ├── config.py       # Paper-aligned learner defaults
│   ├── contracts.py    # Context, outcome, direction, and semantic contracts
│   ├── evidence.py     # Qualification and monitoring ledgers
│   ├── learning.py     # Directional learner lifecycle
│   ├── model.py        # Conditional response model and replay buffer
│   ├── providers.py    # GPT, Claude, and Gemini interfaces
│   └── runtime.py      # Frozen readout and host boundary
├── pyproject.toml
└── requirements.txt
```

## Dataset processing

Input records must contain an instance identity, direction, split, public context, and legal response set. Complete, unknown, and truncated responses are supported. Missing responses remain in the processed output and are not replaced with model predictions.

```bash
PYTHONPATH=. python -m dyadic_tacit.datasets.process overcooked \
  --input /path/to/overcooked.jsonl \
  --output /path/to/processed/overcooked

PYTHONPATH=. python -m dyadic_tacit.datasets.process hanabi \
  --input /path/to/hanabi.jsonl \
  --output /path/to/processed/hanabi

PYTHONPATH=. python -m dyadic_tacit.datasets.process coblock \
  --input /path/to/coblock.jsonl \
  --output /path/to/processed/coblock
```

Each processor writes `opportunities.jsonl`, `instances.jsonl`, and `manifest.json`, including canonical response labels, usage labels, semantic mapping hashes, and source provenance.

## Learner training

Train both independent directions from processed formation records:

```bash
PYTHONPATH=. python -m dyadic_tacit.datasets.learner hanabi \
  --input /path/to/processed/hanabi \
  --output /path/to/learner/hanabi \
  --split formation
```

Default learner settings are AdamW, learning rate `1e-3`, weight decay `1e-4`, four optimizer steps per instance, gradient clipping at `1.0`, replay capacity `512`, at most four records per instance-key, at most eight historical instances per update, warmup `W=8`, and prospective validation blocks of length `L_val=16`.

