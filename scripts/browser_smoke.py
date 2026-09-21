"""Headless browser verification for the live local dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def main() -> None:
    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto("http://127.0.0.1:8765")
        page.wait_for_load_state("networkidle")
        page.locator("#live-label").wait_for(state="visible")
        page.wait_for_function("document.querySelector('#live-label').textContent === 'LIVE LOCAL'")
        page.wait_for_function(
            "Number(document.querySelector('#steps').textContent) >= 3", timeout=45_000
        )

        assert page.locator("#board").is_visible()
        assert page.locator('[data-direction="UP"] .prob-value').text_content() != "—"
        assert page.locator("#device").text_content() == "CUDA"

        page.locator("#pause-button").click()
        page.wait_for_function(
            "document.querySelector('#pause-button').textContent.includes('RESUME')"
        )
        paused_step = page.locator("#steps").text_content()
        page.wait_for_timeout(500)
        assert page.locator("#steps").text_content() == paused_step

        shield = page.locator("#shield-toggle")
        shield.uncheck()
        page.wait_for_timeout(100)
        assert not shield.is_checked()

        seed_before = int(page.locator("#seed").text_content())
        page.locator("#new-button").click()
        page.wait_for_function(
            f"Number(document.querySelector('#seed').textContent) === {seed_before + 1}"
        )
        page.wait_for_function("Number(document.querySelector('#steps').textContent) >= 1")
        page.screenshot(path=ARTIFACTS / "web-dashboard-desktop.png", full_page=True)

        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(200)
        assert (
            page.locator(".workspace").evaluate("el => getComputedStyle(el).gridTemplateColumns")
            == "370px"
        )
        page.screenshot(path=ARTIFACTS / "web-dashboard-mobile.png", full_page=True)

        result = {
            "title": page.title(),
            "seed": page.locator("#seed").text_content(),
            "steps": page.locator("#steps").text_content(),
            "device": page.locator("#device").text_content(),
            "console_errors": console_errors,
            "page_errors": page_errors,
        }
        print(json.dumps(result, indent=2))
        assert not console_errors
        assert not page_errors
        browser.close()


if __name__ == "__main__":
    main()
