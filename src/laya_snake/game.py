"""Deterministic Snake game state with no rendering dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Final

Point = tuple[int, int]

DIRECTIONS: Final[dict[str, Point]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def add_point(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def hamiltonian_cycle(width: int, height: int) -> tuple[Point, ...]:
    """Return a rectangular-grid Hamiltonian cycle.

    At least one dimension must be even. The returned sequence contains every
    board cell once, with adjacent consecutive cells and an adjacent wraparound.
    """

    if width < 2 or height < 2:
        raise ValueError("board dimensions must both be at least 2")
    if width % 2 and height % 2:
        raise ValueError("at least one board dimension must be even")

    if width % 2:
        transposed = hamiltonian_cycle(height, width)
        return tuple((y, x) for x, y in transposed)

    cells: list[Point] = [(0, y) for y in range(height)]
    for x in range(1, width):
        ys = range(height - 1, 0, -1) if x % 2 else range(1, height)
        cells.extend((x, y) for y in ys)
    cells.extend((x, 0) for x in range(width - 1, 0, -1))
    return tuple(cells)


@dataclass(frozen=True)
class StepResult:
    alive: bool
    ate_food: bool
    won: bool
    reason: str | None = None


class SnakeGame:
    """Mutable Snake environment initialized in Hamiltonian-cycle order."""

    def __init__(
        self,
        width: int = 24,
        height: int = 16,
        initial_length: int = 6,
        seed: int = 7,
    ) -> None:
        self.width = width
        self.height = height
        self.initial_length = initial_length
        self.seed = seed
        self.cycle = hamiltonian_cycle(width, height)
        self.cycle_index = {cell: i for i, cell in enumerate(self.cycle)}
        if not 2 <= initial_length < len(self.cycle):
            raise ValueError("initial_length must be between 2 and board area - 1")
        self._rng = Random(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        self.snake: list[Point] = [self.cycle[i] for i in range(self.initial_length - 1, -1, -1)]
        self.score = 0
        self.steps = 0
        self.alive = True
        self.won = False
        self.death_reason: str | None = None
        self.food = self._spawn_food()

    @property
    def head(self) -> Point:
        return self.snake[0]

    @property
    def tail(self) -> Point:
        return self.snake[-1]

    @property
    def heading(self) -> str:
        neck = self.snake[1]
        delta = self.head[0] - neck[0], self.head[1] - neck[1]
        return next((name for name, vector in DIRECTIONS.items() if vector == delta), "UNKNOWN")

    def in_bounds(self, point: Point) -> bool:
        return 0 <= point[0] < self.width and 0 <= point[1] < self.height

    def destination(self, action: str) -> Point:
        try:
            return add_point(self.head, DIRECTIONS[action])
        except KeyError as exc:
            raise ValueError(f"unknown action: {action}") from exc

    def is_legal(self, action: str) -> bool:
        destination = self.destination(action)
        if not self.in_bounds(destination):
            return False
        eating = destination == self.food
        occupied = self.snake if eating else self.snake[:-1]
        return destination not in occupied

    def step(self, action: str) -> StepResult:
        if not self.alive:
            raise RuntimeError("cannot step a finished game")
        destination = self.destination(action)
        if not self.is_legal(action):
            self.alive = False
            self.death_reason = (
                "wall collision" if not self.in_bounds(destination) else "body collision"
            )
            return StepResult(False, False, False, self.death_reason)

        ate_food = destination == self.food
        self.snake.insert(0, destination)
        self.steps += 1
        if ate_food:
            self.score += 1
            if len(self.snake) == self.width * self.height:
                self.won = True
                self.alive = False
                return StepResult(False, True, True, "board filled")
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        return StepResult(True, ate_food, False)

    def _spawn_food(self) -> Point:
        occupied = set(self.snake)
        available = [cell for cell in self.cycle if cell not in occupied]
        if not available:
            return self.head
        return self._rng.choice(available)
