"""Rich terminal dashboard and lightweight Windows keyboard polling."""

from __future__ import annotations

import os
from dataclasses import dataclass

from rich.align import Align
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from .game import SnakeGame
from .planner import Plan
from .policy import Decision


def _bar(value: float, width: int = 18) -> str:
    filled = min(width, max(0, round(value * width)))
    return "█" * filled + "░" * (width - filled)


def _board(game: SnakeGame) -> Panel:
    head = game.head
    body = set(game.snake[1:])
    lines: list[Text] = []
    border = Text("┌" + "──" * game.width + "┐", style="bright_black")
    lines.append(border)
    for y in range(game.height):
        line = Text("│", style="bright_black")
        for x in range(game.width):
            cell = x, y
            if cell == head:
                line.append("██", style="bold bright_green")
            elif cell in body:
                line.append("██", style="green")
            elif cell == game.food:
                line.append("● ", style="bold bright_red")
            else:
                line.append("· ", style="grey23")
        line.append("│", style="bright_black")
        lines.append(line)
    lines.append(Text("└" + "──" * game.width + "┘", style="bright_black"))
    subtitle = f"score {game.score}  length {len(game.snake)}  step {game.steps}"
    return Panel(Group(*lines), title="SNAKE", subtitle=subtitle, border_style="green")


def _metrics(
    decision: Decision | None,
    plan: Plan,
    executed: str | None,
    shielded: bool,
    device: str,
    rate: float,
    interventions: int,
    paused: bool,
    backend: str,
) -> Panel:
    content: list = []
    title = Text(f"{backend.upper()} decision", style="bold cyan")
    if paused:
        title.append("  PAUSED", style="bold yellow")
    content.append(title)
    content.append(Text())
    if decision is None:
        content.append(Text("Waiting for first inference…", style="dim"))
    else:
        probabilities = decision.probabilities
        for action in ("UP", "DOWN", "LEFT", "RIGHT"):
            value = probabilities.get(action, 0.0)
            safe = "SAFE" if action in plan.safe_actions else "----"
            style = "bright_green" if action == executed else "white"
            content.append(
                Text.assemble(
                    (f"{action:<5} ", style),
                    (_bar(value), "cyan"),
                    (f" {value:>6.1%}  {safe}", "green" if safe == "SAFE" else "dim"),
                )
            )
        content.extend(
            [
                Text(),
                Text.assemble(("RAW CHOICE       ", "dim"), (decision.raw_action, "bold white")),
                Text.assemble(
                    ("EXECUTED         ", "dim"),
                    (executed or "—", "bold bright_green"),
                    ("  SHIELD" if shielded else "", "bold yellow"),
                ),
                Text.assemble(
                    ("DEAD-END RISK    ", "dim"),
                    (f"{decision.dead_end_risk:.1%}", "yellow"),
                ),
                Text.assemble(
                    ("FOOD REACHABLE   ", "dim"),
                    (f"{decision.food_reachable_probability:.1%}", "bright_green"),
                ),
                Text.assemble(
                    ("INFERENCE        ", "dim"),
                    (f"{decision.latency_ms:.1f} ms", "cyan"),
                ),
                Text.assemble(("INPUT TOKENS     ", "dim"), (str(decision.input_tokens), "white")),
            ]
        )
    content.extend(
        [
            Text(),
            Text.assemble(("DEVICE           ", "dim"), (device.upper(), "magenta")),
            Text.assemble(("MODEL RATE       ", "dim"), (f"{rate:.2f}/s", "cyan")),
            Text.assemble(("INTERVENTIONS    ", "dim"), (str(interventions), "yellow")),
            Text.assemble(("LOCAL MODEL      ", "dim"), ("OFFLINE", "bold green")),
            Text(),
            Text("Space pause · ↑/↓ speed · R reset · Q quit", style="dim"),
        ]
    )
    return Panel(Group(*content), title="DECISION ENGINE", border_style="cyan", width=49)


def dashboard(
    game: SnakeGame,
    plan: Plan,
    decision: Decision | None,
    executed: str | None,
    shielded: bool,
    device: str,
    rate: float,
    interventions: int,
    paused: bool = False,
    backend: str = "laya",
):
    header = Align.center(
        Text(
            f"{backend.upper()} SNAKE CUDA  ·  every move is a fresh local model decision",
            style="bold white",
        )
    )
    columns = Columns(
        [
            _board(game),
            _metrics(
                decision,
                plan,
                executed,
                shielded,
                device,
                rate,
                interventions,
                paused,
                backend,
            ),
        ],
        padding=(0, 1),
        expand=True,
    )
    return Group(header, Text(), columns)


@dataclass(frozen=True)
class KeyEvent:
    name: str


def poll_keys() -> list[KeyEvent]:
    """Return currently buffered controls without blocking the game loop."""

    if os.name != "nt":
        return []
    import msvcrt

    events: list[KeyEvent] = []
    while msvcrt.kbhit():
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            if code == "H":
                events.append(KeyEvent("faster"))
            elif code == "P":
                events.append(KeyEvent("slower"))
        elif char == " ":
            events.append(KeyEvent("pause"))
        elif char.lower() == "q":
            events.append(KeyEvent("quit"))
        elif char.lower() == "r":
            events.append(KeyEvent("reset"))
        elif char in ("+", "="):
            events.append(KeyEvent("faster"))
        elif char in ("-", "_"):
            events.append(KeyEvent("slower"))
    return events
