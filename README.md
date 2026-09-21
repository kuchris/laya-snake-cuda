# Laya / SemIf Snake CUDA

A local, real-time Snake decision demo with switchable Laya and SemIf typed-decision backends on
NVIDIA CUDA. Every move performs a fresh model inference. No Snake training or cloud API is used.

![Laya Snake monochrome web dashboard](docs/assets/laya-snake-dashboard.png)

The deterministic planner supplies identical compact spatial features to either backend. The model
returns probabilities for all four directions plus two estimates. By default, a Hamiltonian-cycle
safety shield prevents an unsafe proposal from being executed; `--unassisted` exposes the raw
policy.

## What is shown

- Live Snake board, score, length, and step count
- Laya or SemIf probabilities for `UP`, `DOWN`, `LEFT`, and `RIGHT`
- Raw model choice versus executed action
- Visible safety-shield interventions
- Dead-end-risk and food-reachability model answers
- Measured inference latency, decision rate, device, and token count
- Optional offline JSONL recording

## Local model layout

The CLI searches `models/` for a checkpoint and never downloads during play:

```text
models/
├── laya-upstream/
│   └── multilingual/
│       ├── model.safetensors
│       ├── rl_agent_config.json
│       ├── encoder/
│       └── tokenizer/
└── semif-upstream/
    └── qwen3.5-4b/
        ├── config.json
        ├── model-00001-of-00002.safetensors
        ├── model-00002-of-00002.safetensors
        └── tokenizer.json
```

## Setup

Requirements: Windows, an NVIDIA GPU with a current driver, and `uv`.

```powershell
uv sync --extra dev
uv run hf download convaiinnovations/laya `
  --include "multilingual/*" `
  --local-dir models/laya-upstream

# Optional SemIf backend (pinned upstream code and pinned Qwen weights)
uv sync --extra dev --extra semif
uv run hf download Qwen/Qwen3.5-4B `
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a `
  --local-dir models/semif-upstream/qwen3.5-4b
```

## Run

```powershell
# Recommended: local web dashboard with a real browser GUI
uv run laya-snake-web

# Run the same dashboard using SemIf/Qwen3.5-4B
uv run --extra semif laya-snake-web --backend semif

# Terminal dashboard at 8 decisions per second
uv run laya-snake

# Run as fast as inference allows
uv run laya-snake --max-speed

# Select either inference backend
uv run laya-snake --backend laya
uv run --extra semif laya-snake --backend semif

# Execute raw top-1 Laya choices without the safety shield
uv run laya-snake --unassisted

# Reproducible finite benchmark with an append-only decision record
uv run laya-snake --headless --max-speed --steps 600 `
  --record artifacts/run-seed-7.jsonl
```

## Compare backends

The benchmark loads one backend at a time and runs identical deterministic seeds. Its JSON artifact
contains per-seed and aggregate score, deaths, safety-shield interventions, raw-safe rate, planner
agreement, mean latency, and p95 latency.

For this three-question workload, the SemIf adapter uses a compact decision prompt and scores all
three prompts in one padded CUDA forward pass. This avoids the extra prefix-prefill pass that is
useful for SemIf workloads with many questions or a long shared state, but slower for this small
fixed batch.

```powershell
# Quick smoke comparison
uv run --extra semif laya-snake-benchmark --steps 20 --seeds 101

# Longer comparison; omit --unassisted to keep the same safety shield for both models
uv run --extra semif laya-snake-benchmark `
  --backends laya semif --steps 600 --seeds 101 102 103 `
  --output artifacts/laya-vs-semif.json
```

Controls: `Space` pauses, `Up/Down` or `+/-` changes speed, `R` resets with the next seed, and
`Q` exits. A terminal around 105 columns wide shows the board and metrics side by side; narrower
terminals automatically stack the panels. The first inference is a warm-up and is normally slower.

The web dashboard runs entirely on `http://127.0.0.1:8765`. The Python backend keeps the selected
local checkpoint resident on CUDA and pushes each completed decision to the browser over a local
WebSocket. The page includes pause/resume, new round, speed, and safety-shield controls; it does
not upload game state or model data.

## Verify

```powershell
uv run pytest -q
uv run ruff check .
uv run python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.is_available())"
```

The model outputs are not calibrated Snake death probabilities. SemIf option scores are conditional
on the supplied options. The planner provides explicit spatial facts; this project demonstrates
local typed decisions and safe execution, not unaided visual board reasoning or reinforcement
learning. Benchmark results compare this Snake workload only and do not establish a general model
ranking.
