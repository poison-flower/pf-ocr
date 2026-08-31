#!/usr/bin/env python3
"""
Batch-OCR for scanned pages of a Japanese light novel (vertical text) via any
OpenAI-compatible API (OpenRouter, a self-hosted proxy, etc.).

This is the recommended OCR backend for this project: it lets a multimodal
LLM "read" a page image directly, including vertical Japanese text, without
any column-segmentation preprocessing.

Usage (after filling in config.json):
    python openrouter_ocr.py --input ./pages --output ./out

Requirements:
    pip install openai pillow natsort tqdm

One-time setup:
    1. Copy config.example.json -> config.json
    2. Fill in your api_key, base_url and model
    3. Optionally edit prompt.txt to fit your book / house style
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

from natsort import natsorted
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Failed to parse {config_path}: {e}", file=sys.stderr)
        sys.exit(1)


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    return prompt_path.read_text(encoding="utf-8").strip()


def image_to_data_url(path: Path, max_dim: int = 2200) -> str:
    """Downscale (if needed) and encode an image as a base64 data URL."""
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def ocr_image(client: OpenAI, model: str, prompt: str, path: Path, retries: int = 3) -> str:
    data_url = image_to_data_url(path)

    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to OCR {path.name} after {retries} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Batch-OCR a light novel via an OpenAI-compatible API")
    parser.add_argument("--input", required=True, help="Folder with scanned page images (jpg/png)")
    parser.add_argument("--output", required=True, help="Folder for the OCR results")
    parser.add_argument(
        "--config", default=str(SCRIPT_DIR / "config.json"),
        help="Path to config.json with api_key/base_url/model (defaults to config.json next to this script)"
    )
    parser.add_argument(
        "--prompt-file", default=str(SCRIPT_DIR / "prompt.txt"),
        help="Path to the prompt file (defaults to prompt.txt next to this script)"
    )
    parser.add_argument("--base-url", default=None, help="Override base_url from config.json")
    parser.add_argument("--model", default=None, help="Override model from config.json")
    parser.add_argument("--api-key", default=None, help="Override api_key from config.json")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="Delay in seconds between requests (useful if you're hitting rate limits)"
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    api_key = args.api_key or config.get("api_key") or os.environ.get("API_KEY")
    base_url = args.base_url or config.get("base_url")
    model = args.model or config.get("model")

    missing = [name for name, val in [("api_key", api_key), ("base_url", base_url), ("model", model)] if not val]
    if missing:
        print(
            f"Missing settings: {', '.join(missing)}. "
            f"Fill them in {args.config} (see config.example.json) or pass --api-key/--base-url/--model.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt = load_prompt(Path(args.prompt_file))

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

    client = OpenAI(base_url=base_url, api_key=api_key)

    combined_path = output_dir / "combined.md"
    failed = []

    with open(combined_path, "w", encoding="utf-8") as combined_f:
        for idx, img_path in enumerate(tqdm(images, desc="OCR"), start=args.start_page):
            txt_out = pages_dir / f"{img_path.stem}.txt"

            if txt_out.exists():
                text = txt_out.read_text(encoding="utf-8")
            else:
                try:
                    text = ocr_image(client, model, prompt, img_path)
                    txt_out.write_text(text, encoding="utf-8")
                except Exception as e:  # noqa: BLE001
                    print(f"\nError on {img_path.name}: {e}", file=sys.stderr)
                    failed.append(img_path.name)
                    text = ""
                    # Do NOT write a file to disk on failure — otherwise the next
                    # run would see the file exists and assume the page is already
                    # done, silently skipping a retry forever.
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
