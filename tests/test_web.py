from pathlib import Path

from laya_snake.web import ASSET_DIR, GameService


def test_web_assets_are_packaged() -> None:
    for name in ("index.html", "style.css", "app.js"):
        path = ASSET_DIR / name
        assert path.is_file()
        assert path.stat().st_size > 100


def test_web_controls_update_local_state_without_loading_model() -> None:
    service = GameService(Path("unused-in-test"), seed=11)

    service.control("speed", 99)
    service.control("shield", False)
    service.control("pause")
    snapshot = service.snapshot()

    assert snapshot["target_fps"] == 30.0
    assert snapshot["shield_enabled"] is False
    assert snapshot["paused"] is True

    service.control("new_game")
    snapshot = service.snapshot()
    assert snapshot["game"]["seed"] == 12
    assert snapshot["paused"] is False
