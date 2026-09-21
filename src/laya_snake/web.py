"""Local FastAPI/WebSocket dashboard for Laya Snake."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import time
import webbrowser
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Timer
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .game import SnakeGame
from .planner import Plan, analyze, shield_action
from .policy import Decision, LayaPolicy, discover_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = Path(__file__).resolve().parent / "web_assets"


class GameService:
    """Own the model and game loop while browsers remain disposable clients."""

    def __init__(self, model_path: Path, device: str = "cuda", seed: int = 7) -> None:
        self.model_path = model_path
        self.requested_device = device
        self.game = SnakeGame(seed=seed)
        self.policy: LayaPolicy | None = None
        self.device = device
        self.model_status = "loading"
        self.error: str | None = None
        self.paused = False
        self.thinking = False
        self.shield_enabled = True
        self.target_fps = 8.0
        self.interventions = 0
        self.last_decision: Decision | None = None
        self.last_plan: Plan | None = None
        self.last_executed: str | None = None
        self.last_shielded = False
        self.inference_history: deque[float] = deque(maxlen=60)
        self.version = 0
        self.generation = 0
        self._task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="laya-snake-game-loop")

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _bump(self) -> None:
        self.version += 1

    async def _run(self) -> None:
        try:
            self.policy = await asyncio.to_thread(
                LayaPolicy, self.model_path, self.requested_device
            )
            self.device = self.policy.device
            if self.requested_device == "cuda" and self.device != "cuda":
                raise RuntimeError(f"CUDA requested, but Laya loaded on {self.device}")
            self.model_status = "ready"
            self._bump()
        except (OSError, RuntimeError, ValueError) as exc:
            self.model_status = "error"
            self.error = str(exc)
            self._bump()
            return

        while not self._stopping:
            if self.paused or not self.game.alive:
                await asyncio.sleep(0.05)
                continue

            tick_started = time.perf_counter()
            generation = self.generation
            plan = analyze(self.game)
            self.thinking = True
            self._bump()
            try:
                decision = await asyncio.to_thread(self.policy.decide, self.game, plan)
            except (OSError, RuntimeError, ValueError) as exc:
                self.model_status = "error"
                self.error = str(exc)
                self.thinking = False
                self._bump()
                return

            if generation != self.generation:
                self.thinking = False
                self._bump()
                continue

            if self.shield_enabled:
                executed, shielded = shield_action(
                    decision.raw_action, decision.probabilities, plan
                )
            else:
                executed, shielded = decision.raw_action, False
            if shielded:
                self.interventions += 1
            self.game.step(executed)
            self.last_decision = decision
            self.last_plan = plan
            self.last_executed = executed
            self.last_shielded = shielded
            self.inference_history.append(decision.latency_ms)
            self.thinking = False
            self._bump()

            elapsed = time.perf_counter() - tick_started
            delay = max(0.0, 1.0 / self.target_fps - elapsed)
            await asyncio.sleep(delay)

    def control(self, action: str, value: Any = None) -> None:
        if action == "pause":
            self.paused = not self.paused
            self.generation += 1
        elif action == "new_game":
            self.generation += 1
            self.game.reset(self.game.seed + 1)
            self.interventions = 0
            self.last_decision = None
            self.last_plan = None
            self.last_executed = None
            self.last_shielded = False
            self.inference_history.clear()
            self.paused = False
        elif action == "speed":
            self.target_fps = max(1.0, min(30.0, float(value)))
        elif action == "shield":
            self.shield_enabled = bool(value)
        self._bump()

    def snapshot(self) -> dict[str, Any]:
        decision = self.last_decision
        plan = self.last_plan
        mean_latency = (
            sum(self.inference_history) / len(self.inference_history)
            if self.inference_history
            else 0.0
        )
        return {
            "version": self.version,
            "model_status": self.model_status,
            "error": self.error,
            "device": self.device.upper(),
            "paused": self.paused,
            "thinking": self.thinking,
            "shield_enabled": self.shield_enabled,
            "target_fps": self.target_fps,
            "interventions": self.interventions,
            "game": {
                "width": self.game.width,
                "height": self.game.height,
                "snake": self.game.snake,
                "food": self.game.food,
                "score": self.game.score,
                "length": len(self.game.snake),
                "steps": self.game.steps,
                "seed": self.game.seed,
                "alive": self.game.alive,
                "won": self.game.won,
                "death_reason": self.game.death_reason,
            },
            "decision": (
                {
                    "probabilities": decision.probabilities,
                    "raw_action": decision.raw_action,
                    "executed": self.last_executed,
                    "shielded": self.last_shielded,
                    "safe_actions": list(plan.safe_actions) if plan else [],
                    "dead_end_risk": decision.dead_end_risk,
                    "food_reachable": decision.food_reachable_probability,
                    "latency_ms": decision.latency_ms,
                    "mean_latency_ms": mean_latency,
                    "model_rate": 1000.0 / mean_latency if mean_latency else 0.0,
                    "input_tokens": decision.input_tokens,
                }
                if decision
                else None
            ),
        }


def create_app(service: GameService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await service.start()
        yield
        await service.stop()

    app = FastAPI(title="Laya Snake CUDA", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ASSET_DIR / "index.html")

    @app.get("/api/health")
    async def health():
        return {"ok": True, "model_status": service.model_status}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        last_version = -1
        try:
            while True:
                if service.version != last_version:
                    await websocket.send_json(service.snapshot())
                    last_version = service.version
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                    if message.get("type") == "control":
                        service.control(message.get("action", ""), message.get("value"))
                except TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Laya Snake web dashboard.")
    parser.add_argument("--model", type=Path, help="Local Laya checkpoint directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_path = args.model.resolve() if args.model else discover_model(PROJECT_ROOT)
    service = GameService(model_path, args.device, args.seed)
    app = create_app(service)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
