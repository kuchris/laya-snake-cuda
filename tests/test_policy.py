from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from laya_snake.game import SnakeGame
from laya_snake.planner import analyze
from laya_snake.policy import LayaPolicy, build_questions


class FakeAgent:
    device = SimpleNamespace(type="cuda")

    def predict(self, state: str, questions: dict) -> dict:
        assert "Safe route: yes" in state
        assert set(questions) == {"next_move", "safe_route", "food_reachable"}
        return {
            "answers": {
                "next_move": {
                    "choice": "RIGHT",
                    "probabilities": {
                        "UP": 0.1,
                        "DOWN": 0.2,
                        "LEFT": 0.3,
                        "RIGHT": 0.4,
                    },
                },
                "safe_route": {"noul": 0.9},
                "food_reachable": {"noul": 0.8},
            },
            "usage": {"input_tokens": 123},
        }


def test_questions_describe_every_direction() -> None:
    game = SnakeGame(8, 6, seed=9)
    questions = build_questions(analyze(game))

    assert set(questions["next_move"]["criteria"]) == {"UP", "DOWN", "LEFT", "RIGHT"}
    assert sum("Best" in text for text in questions["next_move"]["criteria"].values()) == 1


def test_laya_policy_parses_typed_answers() -> None:
    game = SnakeGame(8, 6, seed=9)
    policy = LayaPolicy(Path("unused-in-test"), agent=FakeAgent())

    decision = policy.decide(game, analyze(game))

    assert decision.raw_action == "RIGHT"
    assert decision.dead_end_risk == 0.1
    assert decision.food_reachable_probability == 0.8
    assert decision.input_tokens == 123
