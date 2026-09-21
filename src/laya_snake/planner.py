"""Deterministic spatial features and an optional cycle-order safety shield."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

from .game import DIRECTIONS, Point, SnakeGame


@dataclass(frozen=True)
class MoveFeatures:
    destination: Point
    legal: bool
    cycle_safe: bool
    cycle_advance: int | None
    food_cycle_distance_after: int | None
    open_cells: int
    food_reachable: bool
    eats_food: bool


@dataclass(frozen=True)
class Plan:
    moves: dict[str, MoveFeatures]
    safe_actions: tuple[str, ...]
    legal_actions: tuple[str, ...]
    food_reachable_now: bool

    @property
    def recommended_action(self) -> str | None:
        if not self.safe_actions:
            return None
        return min(
            self.safe_actions,
            key=lambda action: (
                self.moves[action].food_cycle_distance_after,
                -self.moves[action].open_cells,
            ),
        )

    def state_dict(self, game: SnakeGame) -> dict:
        return {
            "game": {
                "board": {"width": game.width, "height": game.height},
                "head": game.head,
                "tail": game.tail,
                "food": game.food,
                "heading": game.heading,
                "length": len(game.snake),
                "score": game.score,
            },
            "planner": {
                "legal_actions": list(self.legal_actions),
                "cycle_safe_actions": list(self.safe_actions),
                "recommended_action": self.recommended_action,
                "food_reachable_now": self.food_reachable_now,
                "moves": {name: asdict(features) for name, features in self.moves.items()},
            },
        }


def forward_distance(start: int, end: int, size: int) -> int:
    return (end - start) % size


def _reachable_cells(game: SnakeGame, start: Point, action: str | None = None) -> set[Point]:
    occupied = set(game.snake)
    if action is not None and game.is_legal(action):
        destination = game.destination(action)
        eating = destination == game.food
        if not eating:
            occupied.discard(game.tail)
        occupied.discard(destination)
        start = destination
    else:
        occupied.discard(start)

    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            neighbor = x + dx, y + dy
            if game.in_bounds(neighbor) and neighbor not in occupied and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def analyze(game: SnakeGame) -> Plan:
    size = len(game.cycle)
    head_index = game.cycle_index[game.head]
    tail_distance = forward_distance(head_index, game.cycle_index[game.tail], size)
    food_distance = forward_distance(head_index, game.cycle_index[game.food], size)
    moves: dict[str, MoveFeatures] = {}

    for action in DIRECTIONS:
        destination = game.destination(action)
        legal = game.is_legal(action)
        if not legal:
            moves[action] = MoveFeatures(destination, False, False, None, None, 0, False, False)
            continue

        destination_index = game.cycle_index[destination]
        advance = forward_distance(head_index, destination_index, size)
        cycle_safe = 0 < advance <= food_distance and advance <= tail_distance
        reachable = _reachable_cells(game, destination, action)
        moves[action] = MoveFeatures(
            destination=destination,
            legal=True,
            cycle_safe=cycle_safe,
            cycle_advance=advance,
            food_cycle_distance_after=forward_distance(
                destination_index, game.cycle_index[game.food], size
            ),
            open_cells=len(reachable),
            food_reachable=game.food in reachable or destination == game.food,
            eats_food=destination == game.food,
        )

    safe_actions = tuple(name for name, move in moves.items() if move.cycle_safe)
    legal_actions = tuple(name for name, move in moves.items() if move.legal)
    reachable_now = game.food in _reachable_cells(game, game.head)
    return Plan(moves, safe_actions, legal_actions, reachable_now)


def shield_action(raw_action: str, probabilities: dict[str, float], plan: Plan) -> tuple[str, bool]:
    """Select the highest-probability admissible action when the raw choice is unsafe."""

    if raw_action in plan.safe_actions:
        return raw_action, False
    candidates = plan.safe_actions or plan.legal_actions
    if not candidates:
        return raw_action, False
    selected = max(candidates, key=lambda action: probabilities.get(action, 0.0))
    return selected, selected != raw_action
