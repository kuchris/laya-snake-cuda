"""Reproducible Laya versus SemIf Snake benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from rich.console import Console
from rich.table import Table

from .game import SnakeGame
from .planner import analyze, shield_action
from .policy import DecisionPolicy, discover_model, load_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def run_seed(
    policy: DecisionPolicy,
    seed: int,
    steps: int,
    *,
    shield_enabled: bool,
    width: int = 24,
    height: int = 16,
    initial_length: int = 6,
) -> dict[str, Any]:
    game = SnakeGame(width, height, initial_length, seed)
    latencies: list[float] = []
    interventions = 0
    raw_safe = 0
    planner_agreements = 0

    while game.alive and len(latencies) < steps:
        plan = analyze(game)
        decision = policy.decide(game, plan)
        latencies.append(decision.latency_ms)
        raw_safe += int(decision.raw_action in plan.safe_actions)
        planner_agreements += int(decision.raw_action == plan.recommended_action)
        if shield_enabled:
            executed, shielded = shield_action(
                decision.raw_action, decision.probabilities, plan
            )
        else:
            executed, shielded = decision.raw_action, False
        interventions += int(shielded)
        game.step(executed)

    steady = latencies[1:] or latencies
    status = "won" if game.won else "alive" if game.alive else game.death_reason or "dead"
    return {
        "seed": seed,
        "decisions": len(latencies),
        "score": game.score,
        "length": len(game.snake),
        "status": status,
        "shield_interventions": interventions,
        "raw_safe_rate": raw_safe / len(latencies) if latencies else 0.0,
        "planner_agreement_rate": planner_agreements / len(latencies) if latencies else 0.0,
        "warmup_latency_ms": latencies[0] if latencies else 0.0,
        "mean_latency_ms": mean(steady) if steady else 0.0,
        "p50_latency_ms": median(steady) if steady else 0.0,
        "p95_latency_ms": _percentile(steady, 0.95),
    }


def summarize(backend: str, model_path: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "backend": backend,
        "model_path": str(model_path),
        "seeds": [run["seed"] for run in runs],
        "runs": runs,
        "aggregate": {
            "mean_score": mean(run["score"] for run in runs),
            "total_decisions": sum(run["decisions"] for run in runs),
            "deaths": sum(run["status"] not in {"alive", "won"} for run in runs),
            "shield_interventions": sum(run["shield_interventions"] for run in runs),
            "mean_raw_safe_rate": mean(run["raw_safe_rate"] for run in runs),
            "mean_planner_agreement_rate": mean(
                run["planner_agreement_rate"] for run in runs
            ),
            "mean_latency_ms": mean(run["mean_latency_ms"] for run in runs),
            "mean_p95_latency_ms": mean(run["p95_latency_ms"] for run in runs),
        },
    }


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Laya and SemIf on identical deterministic Snake runs."
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("laya", "semif"),
        default=("laya", "semif"),
    )
    parser.add_argument("--laya-model", type=Path)
    parser.add_argument("--semif-model", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=(101, 102, 103))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--unassisted", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _model_path(args: argparse.Namespace, backend: str) -> Path:
    supplied = getattr(args, f"{backend}_model")
    return supplied.resolve() if supplied else discover_model(PROJECT_ROOT, backend)


def _print_summary(console: Console, results: list[dict[str, Any]]) -> None:
    table = Table(title="Snake backend benchmark")
    for heading in (
        "Backend",
        "Mean score",
        "Deaths",
        "Shield cuts",
        "Raw safe",
        "Planner agree",
        "Mean ms",
        "Mean p95 ms",
    ):
        table.add_column(heading)
    for result in results:
        item = result["aggregate"]
        table.add_row(
            result["backend"].upper(),
            f"{item['mean_score']:.2f}",
            str(item["deaths"]),
            str(item["shield_interventions"]),
            f"{item['mean_raw_safe_rate']:.1%}",
            f"{item['mean_planner_agreement_rate']:.1%}",
            f"{item['mean_latency_ms']:.1f}",
            f"{item['mean_p95_latency_ms']:.1f}",
        )
    console.print(table)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    console = Console()
    if args.steps < 1:
        console.print("[bold red]Error:[/] --steps must be at least 1", file=sys.stderr)
        raise SystemExit(2)

    results: list[dict[str, Any]] = []
    try:
        for backend in args.backends:
            model_path = _model_path(args, backend)
            with console.status(
                f"Loading {backend.upper()} from {model_path} on {args.device.upper()}…"
            ):
                policy = load_policy(backend, model_path, args.device)
            runs = []
            for seed in args.seeds:
                console.print(f"[dim]{backend.upper()} · seed {seed} · {args.steps} moves[/]")
                runs.append(
                    run_seed(
                        policy,
                        seed,
                        args.steps,
                        shield_enabled=not args.unassisted,
                    )
                )
            results.append(summarize(backend, model_path, runs))
            del policy
            _release_cuda()
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    output = args.output
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        output = PROJECT_ROOT / "artifacts" / f"backend-benchmark-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "shield_enabled": not args.unassisted,
        "steps_per_seed": args.steps,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_summary(console, results)
    console.print(f"Results: [bold]{output}[/]")


if __name__ == "__main__":
    main()
