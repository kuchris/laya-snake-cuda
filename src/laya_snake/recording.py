"""Append-only JSONL recording for deterministic inspection and replay tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self, TextIO

from .game import SnakeGame
from .planner import Plan
from .policy import Decision


class Recorder:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._file: TextIO | None = None

    def __enter__(self) -> Self:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("x", encoding="utf-8", newline="\n")
        return self

    def write(
        self,
        game: SnakeGame,
        plan: Plan,
        decision: Decision,
        executed: str,
        shielded: bool,
    ) -> None:
        if self._file is None:
            return
        payload = {
            "seed": game.seed,
            "step": game.steps,
            "score": game.score,
            "snake": game.snake,
            "food": game.food,
            "planner": plan.state_dict(game)["planner"],
            "decision": {
                "raw_action": decision.raw_action,
                "probabilities": decision.probabilities,
                "dead_end_risk": decision.dead_end_risk,
                "food_reachable_probability": decision.food_reachable_probability,
                "latency_ms": decision.latency_ms,
                "input_tokens": decision.input_tokens,
            },
            "executed": executed,
            "shielded": shielded,
        }
        self._file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._file.flush()

    def __exit__(self, *_exc_info) -> None:
        if self._file is not None:
            self._file.close()
