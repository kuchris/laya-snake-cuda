from __future__ import annotations

import pytest

from laya_snake.game import DIRECTIONS, SnakeGame, hamiltonian_cycle


@pytest.mark.parametrize("width,height", [(4, 5), (6, 6), (5, 4), (24, 16)])
def test_hamiltonian_cycle_covers_board_with_adjacent_wraparound(width: int, height: int) -> None:
    cycle = hamiltonian_cycle(width, height)

    assert len(cycle) == width * height
    assert len(set(cycle)) == len(cycle)
    for current, following in zip(cycle, cycle[1:] + cycle[:1]):
        distance = abs(current[0] - following[0]) + abs(current[1] - following[1])
        assert distance == 1


def test_odd_by_odd_board_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension must be even"):
        hamiltonian_cycle(5, 5)


def test_cycle_successor_is_a_legal_move() -> None:
    game = SnakeGame(8, 6, initial_length=5, seed=3)
    successor = game.cycle[(game.cycle_index[game.head] + 1) % len(game.cycle)]
    delta = successor[0] - game.head[0], successor[1] - game.head[1]
    action = next(name for name, vector in DIRECTIONS.items() if vector == delta)

    result = game.step(action)

    assert result.alive
    assert game.head == successor
