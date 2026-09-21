"""Command-line entry point for the live Laya Snake demo."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

from rich.console import Console
from rich.live import Live

from .display import dashboard, poll_keys
from .game import SnakeGame
from .planner import analyze, shield_action
from .policy import LayaPolicy, discover_model
from .recording import Recorder

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Snake with a fresh local Laya decision every move."
    )
    parser.add_argument("--model", type=Path, help="Local Laya checkpoint directory")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--initial-length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--max-speed", action="store_true", help="Run as soon as each inference finishes"
    )
    parser.add_argument(
        "--unassisted", action="store_true", help="Execute raw top-1 choices without shield"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Do not draw the terminal dashboard"
    )
    parser.add_argument("--steps", type=int, help="Stop after this many attempted moves")
    parser.add_argument(
        "--record", type=Path, help="Create an append-only JSONL decision recording"
    )
    parser.add_argument(
        "--no-alt-screen", action="store_true", help="Render inline instead of full-screen"
    )
    return parser


def _run(args: argparse.Namespace, console: Console) -> int:
    model_path = args.model.resolve() if args.model else discover_model(PROJECT_ROOT)
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")

    with console.status(f"Loading local Laya checkpoint on {args.device.upper()}…"):
        policy = LayaPolicy(model_path, device=args.device)
    if args.device == "cuda" and policy.device != "cuda":
        raise RuntimeError(f"CUDA was requested, but Laya loaded on {policy.device}")

    game = SnakeGame(args.width, args.height, args.initial_length, args.seed)
    target_fps = max(1.0, args.fps)
    paused = False
    quit_requested = False
    interventions = 0
    decisions = 0
    latencies: deque[float] = deque(maxlen=60)
    last_view = None

    live_context = (
        Live(console=console, screen=not args.no_alt_screen, refresh_per_second=20, transient=False)
        if not args.headless
        else None
    )

    with Recorder(args.record) as recorder:
        if live_context:
            live_context.start()
        try:
            while not quit_requested and game.alive:
                for event in poll_keys():
                    if event.name == "quit":
                        quit_requested = True
                    elif event.name == "pause":
                        paused = not paused
                    elif event.name == "faster":
                        target_fps = min(60.0, target_fps + 2.0)
                    elif event.name == "slower":
                        target_fps = max(1.0, target_fps - 2.0)
                    elif event.name == "reset":
                        game.reset(game.seed + 1)
                        interventions = 0
                if quit_requested:
                    break
                plan = analyze(game)
                if paused:
                    if live_context and last_view is not None:
                        live_context.update(last_view)
                    time.sleep(0.05)
                    continue

                tick_started = time.perf_counter()
                decision = policy.decide(game, plan)
                if args.unassisted:
                    executed, shielded = decision.raw_action, False
                else:
                    executed, shielded = shield_action(
                        decision.raw_action, decision.probabilities, plan
                    )
                if shielded:
                    interventions += 1
                recorder.write(game, plan, decision, executed, shielded)
                decisions += 1
                latencies.append(decision.latency_ms)
                rate = 1000.0 / (sum(latencies) / len(latencies)) if latencies else 0.0
                last_view = dashboard(
                    game,
                    plan,
                    decision,
                    executed,
                    shielded,
                    policy.device,
                    rate,
                    interventions,
                    paused,
                )
                if live_context:
                    live_context.update(last_view, refresh=True)
                game.step(executed)

                if args.steps is not None and decisions >= args.steps:
                    break
                if not args.max_speed:
                    remaining = 1.0 / target_fps - (time.perf_counter() - tick_started)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            if live_context:
                if last_view is not None:
                    live_context.update(last_view, refresh=True)
                live_context.stop()

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    status = "won" if game.won else "alive" if game.alive else game.death_reason or "stopped"
    console.print(
        f"[bold green]Run complete[/]: {decisions} decisions, score {game.score}, "
        f"length {len(game.snake)}, {interventions} shield interventions, "
        f"mean recent inference {mean_latency:.1f} ms, status {status}."
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()
    try:
        raise SystemExit(_run(args, console))
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
