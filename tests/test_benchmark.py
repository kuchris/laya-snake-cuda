from __future__ import annotations

from laya_snake.benchmark import run_seed, summarize
from laya_snake.policy import Decision


class PlannerFollowingPolicy:
    name = "fake"
    model_label = "FAKE"
    device = "cpu"

    def decide(self, _game, plan):
        action = plan.recommended_action
        probabilities = {name: 0.0 for name in plan.moves}
        probabilities[action] = 1.0
        return Decision(action, probabilities, 0.0, 1.0, 1.0, 10)


def test_benchmark_is_reproducible_for_same_seed() -> None:
    policy = PlannerFollowingPolicy()

    first = run_seed(policy, 17, 20, shield_enabled=True, width=8, height=6)
    second = run_seed(policy, 17, 20, shield_enabled=True, width=8, height=6)

    assert first == second
    assert first["decisions"] == 20
    assert first["raw_safe_rate"] == 1.0
    assert first["planner_agreement_rate"] == 1.0


def test_benchmark_summary_aggregates_runs() -> None:
    policy = PlannerFollowingPolicy()
    runs = [
        run_seed(policy, seed, 10, shield_enabled=True, width=8, height=6)
        for seed in (1, 2)
    ]

    result = summarize("fake", __import__("pathlib").Path("fake-model"), runs)

    assert result["aggregate"]["total_decisions"] == 20
    assert result["aggregate"]["mean_latency_ms"] == 1.0
