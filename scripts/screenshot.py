"""
Capture a screenshot of the Expert Lens UI for the README.

Requires: pip install playwright && playwright install chromium
"""

import asyncio
import os
import sys

async def main():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Install playwright first: pip install playwright && playwright install chromium")
        sys.exit(1)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "screenshots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo.png")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 900})

        await page.goto("http://localhost:5173", wait_until="networkidle")
        await asyncio.sleep(1)

        # Click an example prompt to trigger inference
        example_btn = page.locator("button", has_text="Once upon a time")
        if await example_btn.count() > 0:
            await example_btn.click()
            await asyncio.sleep(3)

        await page.screenshot(path=out_path, full_page=True)
        print(f"Screenshot saved to {out_path}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
