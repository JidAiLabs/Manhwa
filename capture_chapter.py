#!/usr/bin/env python3
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image

def wait_images_loaded(page, timeout_ms=15000):
    # Wait until all images currently in DOM are loaded
    page.wait_for_function(
        """() => {
            const imgs = Array.from(document.images || []);
            if (imgs.length === 0) return true;
            return imgs.every(img => img.complete && img.naturalWidth > 0);
        }""",
        timeout=timeout_ms
    )

def stable_scroll_height(page, rounds=6, delay_ms=400):
    # Wait until scrollHeight stops increasing (lazy load stabilization)
    prev = page.evaluate("document.body.scrollHeight")
    stable = 0
    while stable < rounds:
        page.wait_for_timeout(delay_ms)
        cur = page.evaluate("document.body.scrollHeight")
        if cur == prev:
            stable += 1
        else:
            stable = 0
            prev = cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="out/raw")
    ap.add_argument("--name", default="chapter_001")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--step", type=int, default=750)      # scroll step (px)
    ap.add_argument("--overlap", type=int, default=150)   # overlap between tiles
    ap.add_argument("--delay", type=int, default=350)     # ms between scrolls
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    tiles_dir = out_dir / f"{args.name}_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    stitched_path = out_dir / f"{args.name}.png"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_timeout(1200)

        # initial load settle
        wait_images_loaded(page, timeout_ms=20000)

        # Scroll down gradually, capturing tiles
        y = 0
        tile_paths = []
        idx = 1

        # Ensure we start at the top
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)

        while True:
            # Wait for any lazy-loaded images at current position
            wait_images_loaded(page, timeout_ms=20000)

            tile_path = tiles_dir / f"tile_{idx:04d}.png"
            page.screenshot(path=str(tile_path), full_page=False)
            tile_paths.append(tile_path)
            idx += 1

            # Compute next scroll position
            scroll_height = page.evaluate("document.body.scrollHeight")
            viewport_h = args.height
            max_y = max(0, scroll_height - viewport_h)

            if y >= max_y:
                break

            y_next = min(max_y, y + args.step)
            page.evaluate(f"window.scrollTo(0, {y_next})")
            page.wait_for_timeout(args.delay)
            y = y_next

            # If page is still expanding due to lazy load, let it stabilize a bit
            stable_scroll_height(page, rounds=3, delay_ms=250)

        browser.close()

    # Stitch tiles vertically (crop overlap to avoid duplicate bands)
    images = [Image.open(p) for p in tile_paths]
    w = images[0].width

    stitched_parts = []
    for i, im in enumerate(images):
        if i == 0:
            stitched_parts.append(im)
        else:
            stitched_parts.append(im.crop((0, args.overlap, w, im.height)))

    total_h = sum(im.height for im in stitched_parts)
    canvas = Image.new("RGB", (w, total_h))

    y_off = 0
    for im in stitched_parts:
        canvas.paste(im, (0, y_off))
        y_off += im.height

    canvas.save(stitched_path)
    print("Saved stitched:", stitched_path)
    print("Tiles in:", tiles_dir)

if __name__ == "__main__":
    main()
