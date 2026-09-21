from __future__ import annotations

from laya_snake.game import SnakeGame
from laya_snake.planner import analyze, shield_action


def test_planner_always_exposes_a_safe_successor_during_shielded_run() -> None:
    game = SnakeGame(8, 6, initial_length=6, seed=11)
    probabilities = {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25}

    for _ in range(3_000):
        if game.won:
            break
        plan = analyze(game)
        assert plan.safe_actions
        assert plan.recommended_action in plan.safe_actions
        action, _ = shield_action("UP", probabilities, plan)
        result = game.step(action)
        assert result.alive or result.won

    assert game.won


def test_shield_replaces_an_unsafe_raw_choice() -> None:
    game = SnakeGame(8, 6, initial_length=6, seed=4)
    plan = analyze(game)
    unsafe = next(action for action in plan.moves if action not in plan.safe_actions)
    probabilities = {name: 0.1 for name in plan.moves}
    probabilities[unsafe] = 0.9

    executed, intervened = shield_action(unsafe, probabilities, plan)

    assert intervened
    assert executed in plan.safe_actions
