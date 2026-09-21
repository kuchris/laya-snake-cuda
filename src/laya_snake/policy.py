"""Laya inference adapter for Snake planner features."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .game import SnakeGame
from .planner import Plan


@dataclass(frozen=True)
class Decision:
    raw_action: str
    probabilities: dict[str, float]
    dead_end_risk: float
    food_reachable_probability: float
    latency_ms: float
    input_tokens: int


class AgentLike(Protocol):
    device: Any

    def predict(self, state: dict, questions: dict) -> dict: ...


def build_questions(plan: Plan) -> dict:
    direction_criteria = {}
    ordered_actions = plan.moves
    for action in ordered_actions:
        move = plan.moves[action]
        if not move.legal:
            direction_criteria[action] = "Blocked. Collision."
        elif not move.cycle_safe:
            direction_criteria[action] = "Unsafe. Traps the snake."
        elif move.eats_food:
            direction_criteria[action] = "Safe. Eat food now. Best."
        elif action == plan.recommended_action:
            direction_criteria[action] = "Safe. Best route to food."
        else:
            direction_criteria[action] = "Safe. Slower route."
    return {
        "next_move": {
            "type": "choice",
            "instructions": "Choose the best safe move toward food.",
            "criteria": direction_criteria,
        },
        "safe_route": {
            "type": "noul",
            "instructions": "Is a safe route available?",
        },
        "food_reachable": {
            "type": "noul",
            "instructions": "Is food reachable through empty cells?",
        },
    }


class LayaPolicy:
    def __init__(
        self, model_path: Path, device: str = "cuda", agent: AgentLike | None = None
    ) -> None:
        self.model_path = model_path
        if agent is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import laya

            agent = laya.load(str(model_path), device=device)
        self.agent = agent

    @property
    def device(self) -> str:
        return str(self.agent.device)

    def decide(self, game: SnakeGame, plan: Plan) -> Decision:
        state = (
            f"Safe route: {'yes' if plan.safe_actions else 'no'}. "
            f"Food reachable through empty cells: {'yes' if plan.food_reachable_now else 'no'}."
        )
        started = time.perf_counter()
        result = self.agent.predict(state, build_questions(plan))
        latency_ms = (time.perf_counter() - started) * 1000
        answers = result["answers"]
        move = answers["next_move"]
        safe_probability = float(answers["safe_route"]["noul"])
        probabilities = {key: float(value) for key, value in move["probabilities"].items()}
        food_probability = float(answers["food_reachable"]["noul"])
        expected_actions = {"UP", "DOWN", "LEFT", "RIGHT"}
        if set(probabilities) != expected_actions:
            raise ValueError("Laya did not return probabilities for all four directions")
        values = [*probabilities.values(), safe_probability, food_probability]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Laya returned an invalid probability; no move was executed")
        return Decision(
            raw_action=str(move["choice"]),
            probabilities=probabilities,
            dead_end_risk=round(1.0 - safe_probability, 4),
            food_reachable_probability=food_probability,
            latency_ms=latency_ms,
            input_tokens=int(result.get("usage", {}).get("input_tokens", 0)),
        )


def discover_model(project_root: Path) -> Path:
    models_dir = project_root / "models"
    candidates = sorted(path.parent for path in models_dir.rglob("rl_agent_config.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No local Laya checkpoint found under {models_dir}. "
            "Expected rl_agent_config.json, model.safetensors, encoder/, and tokenizer/."
        )
    preferred = [path for path in candidates if "multilingual" in path.name.lower()]
    return preferred[0] if preferred else candidates[0]
