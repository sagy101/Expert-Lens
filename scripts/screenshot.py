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
    
    demo_path = os.path.join(out_dir, "demo.png")
    analysis_path = os.path.join(out_dir, "analysis.png")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})

        print("Navigating to app...")
        await page.goto("http://localhost:5173", wait_until="networkidle")
        await asyncio.sleep(2)

        # --- 1. Inference View ---
        print("Running inference demo...")
        # Click an example prompt to trigger inference
        example_btn = page.locator("button", has_text="Once upon a time")
        if await example_btn.count() > 0:
            await example_btn.first.click()
            # Wait for inference to complete (look for tokens appearing)
            await asyncio.sleep(3)
        
        await page.screenshot(path=demo_path, full_page=False)
        print(f"Screenshot saved to {demo_path}")

        # --- 2. Analyzer View ---
        print("Running expert analysis...")
        # Switch to Analyzer tab
        await page.click("text=Expert Analyzer")
        await asyncio.sleep(1)

        # Click Run Analysis
        run_btn = page.locator("button", has_text="Run Analysis")
        await run_btn.click()

        # Wait for analysis steps (ingest -> map -> profile)
        # We added 2.5s delays for steps, plus LLM time. Give it plenty of time.
        # Wait until "Re-analyze" appears
        try:
            await page.wait_for_selector("text=Re-analyze", timeout=60000)
        except Exception as e:
            print(f"Timeout waiting for analysis to complete: {e}")
            # Take a screenshot anyway to debug
            await page.screenshot(path=os.path.join(out_dir, "analysis_error.png"), full_page=False)
            raise e
            
        await asyncio.sleep(1) # Let animations settle

        await page.screenshot(path=analysis_path, full_page=False)
        print(f"Screenshot saved to {analysis_path}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
