"""Typed-decision backends for Snake planner features."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .game import SnakeGame
from .planner import Plan

SEMIF_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


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


class DecisionPolicy(Protocol):
    name: str
    model_label: str

    @property
    def device(self) -> str: ...

    def decide(self, game: SnakeGame, plan: Plan) -> Decision: ...


def build_state(plan: Plan) -> str:
    return (
        f"Safe route: {'yes' if plan.safe_actions else 'no'}. "
        f"Food reachable through empty cells: {'yes' if plan.food_reachable_now else 'no'}."
    )


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
    name = "laya"
    model_label = "MMBERT · 322M"

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
        state = build_state(plan)
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


def build_semif_rows(plan: Plan) -> list[dict[str, Any]]:
    questions = build_questions(plan)
    state = build_state(plan)
    return [
        {
            "id": "next_move",
            "state": state,
            "question": questions["next_move"]["instructions"],
            "options": [
                {"id": action, "description": description}
                for action, description in questions["next_move"]["criteria"].items()
            ],
        },
        {
            "id": "safe_route",
            "state": state,
            "question": questions["safe_route"]["instructions"],
            "options": [
                {"id": "yes", "description": "Yes"},
                {"id": "no", "description": "No"},
            ],
        },
        {
            "id": "food_reachable",
            "state": state,
            "question": questions["food_reachable"]["instructions"],
            "options": [
                {"id": "yes", "description": "Yes"},
                {"id": "no", "description": "No"},
            ],
        },
    ]


SemIfScorer = Callable[[list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, Any]]]


class SemIfPolicy:
    """SemIf shared-state option-logit backend using local Qwen3.5-4B weights."""

    name = "semif"
    model_label = "QWEN3.5 · 4B"

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        revision: str = SEMIF_MODEL_REVISION,
        scorer: SemIfScorer | None = None,
    ) -> None:
        self.model_path = model_path
        self._device = device
        if scorer is None:
            if device != "cuda":
                raise ValueError("The SemIf backend currently requires CUDA")
            try:
                from semif_phase1.core import load_causal_model
            except ImportError as exc:
                raise RuntimeError(
                    "SemIf is not installed. Run `uv sync --extra semif --extra dev`."
                ) from exc

            model, tokenizer, metadata = load_causal_model(str(model_path), revision)
            from .semif_batch import encode_snake_prompt, score_batch

            def scorer(rows: list[dict[str, Any]]):
                return score_batch(
                    model, tokenizer, rows, metadata, encoder=encode_snake_prompt
                )

            self._model = model
            self._tokenizer = tokenizer
        self._scorer = scorer

    @property
    def device(self) -> str:
        return self._device

    def decide(self, game: SnakeGame, plan: Plan) -> Decision:
        started = time.perf_counter()
        results, _timing = self._scorer(build_semif_rows(plan))
        latency_ms = (time.perf_counter() - started) * 1000
        by_id = {result["id"]: result for result in results}
        if set(by_id) != {"next_move", "safe_route", "food_reachable"}:
            raise ValueError("SemIf did not return all three Snake decisions")

        def distribution(result: dict[str, Any]) -> dict[str, float]:
            ids = result.get("option_ids", [])
            values = result.get("probabilities", [])
            if len(ids) != len(values):
                raise ValueError("SemIf returned mismatched option IDs and probabilities")
            return {str(key): float(value) for key, value in zip(ids, values)}

        probabilities = distribution(by_id["next_move"])
        safe = distribution(by_id["safe_route"])
        reachable = distribution(by_id["food_reachable"])
        expected_actions = {"UP", "DOWN", "LEFT", "RIGHT"}
        if set(probabilities) != expected_actions:
            raise ValueError("SemIf did not return probabilities for all four directions")
        values = [*probabilities.values(), *safe.values(), *reachable.values()]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("SemIf returned an invalid probability; no move was executed")
        if set(safe) != {"yes", "no"} or set(reachable) != {"yes", "no"}:
            raise ValueError("SemIf binary decisions must contain yes and no")
        return Decision(
            raw_action=max(probabilities, key=probabilities.get),
            probabilities=probabilities,
            dead_end_risk=round(safe["no"], 4),
            food_reachable_probability=reachable["yes"],
            latency_ms=latency_ms,
            input_tokens=sum(int(result.get("input_tokens", 0)) for result in results),
        )


def discover_model(project_root: Path, backend: str = "laya") -> Path:
    models_dir = project_root / "models"
    if backend == "semif":
        candidates = sorted(
            path.parent
            for path in models_dir.rglob("config.json")
            if any(path.parent.glob("model*.safetensors"))
            and "qwen3.5-4b" in path.parent.name.lower()
        )
        if not candidates:
            raise FileNotFoundError(
                f"No local SemIf Qwen3.5-4B checkpoint found under {models_dir}. "
                "Expected a qwen3.5-4b directory with config.json and model safetensors."
            )
        return candidates[0]
    if backend != "laya":
        raise ValueError(f"Unknown backend: {backend}")
    candidates = sorted(path.parent for path in models_dir.rglob("rl_agent_config.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No local Laya checkpoint found under {models_dir}. "
            "Expected rl_agent_config.json, model.safetensors, encoder/, and tokenizer/."
        )
    preferred = [path for path in candidates if "multilingual" in path.name.lower()]
    return preferred[0] if preferred else candidates[0]


def load_policy(
    backend: str,
    model_path: Path,
    device: str = "cuda",
    semif_revision: str = SEMIF_MODEL_REVISION,
) -> DecisionPolicy:
    if backend == "laya":
        return LayaPolicy(model_path, device=device)
    if backend == "semif":
        return SemIfPolicy(model_path, device=device, revision=semif_revision)
    raise ValueError(f"Unknown backend: {backend}")
