#!/usr/bin/env python3
"""
Batch-OCR for scanned pages of a Japanese light novel (vertical text) via the
Gemini API directly (not through a proxy).

Usage:
    export GEMINI_API_KEY="your_key"
    python gemini_direct_ocr.py --input ./pages --output ./out

Requirements:
    pip install google-genai pillow natsort tqdm

If you access Gemini (or another model) through OpenRouter or a similar
OpenAI-compatible proxy instead of a direct Google API key, use
openrouter_ocr.py instead — it's the recommended entry point for this
project and shares the same prompt.txt / config.json workflow.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

PROMPT = """\
This is a scanned page from a Japanese light novel, printed in vertical text (縦書き).
Transcribe ALL Japanese text on the page, preserving correct reading order
(columns read top-to-bottom, then right-to-left).

Rules:
- Do not translate or summarize — output only the transcribed Japanese text.
- No line numbers, no commentary, no explanations of your own.
- Furigana may be omitted (only the base kanji/kana text is needed).
- If the page has no text at all (illustration-only, blank, cover/technical
  page), output exactly: [NO_TEXT]
- Preserve paragraph breaks where the layout clearly shows them.
"""


def ocr_image(client: genai.Client, model: str, path: Path, retries: int = 3) -> str:
    img = Image.open(path)
    # Downscale very large scans — speeds up and cheapens the request without
    # a noticeable loss of OCR quality.
    max_dim = 2200
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))

    last_err = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[PROMPT, img],
                config=types.GenerateContentConfig(temperature=0),
            )
            text = (response.text or "").strip()
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to OCR {path.name} after {retries} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Batch-OCR a light novel via the Gemini API")
    parser.add_argument("--input", required=True, help="Folder with scanned page images (jpg/png)")
    parser.add_argument("--output", required=True, help="Folder for the OCR results")
    parser.add_argument("--model", default="gemini-3.7-flash", help="Gemini model name")
    parser.add_argument(
        "--api-key", default=None,
        help="API key (falls back to the GEMINI_API_KEY environment variable)"
    )
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="Delay in seconds between requests (useful if you're hitting rate limits)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No API key found. Pass --api-key or set GEMINI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages_txt"
    pages_dir.mkdir(exist_ok=True)

    images = [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    images = natsorted(images, key=lambda p: p.name)

    if not images:
        print(f"No images found in {input_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Pages found: {len(images)}")

    client = genai.Client(api_key=api_key)

    combined_path = output_dir / "combined.md"
    failed = []

    with open(combined_path, "w", encoding="utf-8") as combined_f:
        for idx, img_path in enumerate(tqdm(images, desc="OCR"), start=args.start_page):
            txt_out = pages_dir / f"{img_path.stem}.txt"

            if txt_out.exists():
                text = txt_out.read_text(encoding="utf-8")
            else:
                try:
                    text = ocr_image(client, args.model, img_path)
                    txt_out.write_text(text, encoding="utf-8")
                except Exception as e:  # noqa: BLE001
                    print(f"\nError on {img_path.name}: {e}", file=sys.stderr)
                    failed.append(img_path.name)
                    text = ""
                    # Do NOT write a file to disk on failure — see openrouter_ocr.py
                    # for the reasoning.
                if args.sleep:
                    time.sleep(args.sleep)

            combined_f.write(f"\n\n<!-- page {idx}: {img_path.name} -->\n\n")
            combined_f.write(text)

    print(f"\nDone. Combined file: {combined_path}")
    print(f"Per-page files: {pages_dir}")
    if failed:
        print(f"\nFailed to OCR {len(failed)} page(s):")
        for name in failed:
            print(f"  - {name}")
        print("Re-run the script with the same --output folder — already-done pages will not be redone.")


if __name__ == "__main__":
    main()
