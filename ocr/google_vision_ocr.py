#!/usr/bin/env python3
"""
Batch-OCR for scanned pages of a Japanese light novel (vertical text) via
Google Cloud Vision.

Usage:
    python google_vision_ocr.py --input ./pages --output ./out --credentials ./key.json

Requirements:
    pip install google-cloud-vision natsort tqdm

Google Cloud setup (one-time, ~5-10 minutes):
    1. Go to https://console.cloud.google.com/
    2. Create a project (or use an existing one)
    3. Search for "Vision API" -> Enable
    4. Go to "APIs & Services" -> "Credentials" -> "Create Credentials" -> "Service account"
    5. Create the service account (role can be left unset, or "Editor")
    6. Open the account -> Keys -> Add Key -> JSON -> downloads key.json
    7. Point --credentials at that file

Note: as of writing, Google requires a billing account to be enabled on the
project before the Vision API will respond, even though usage stays within
the free tier (1000 requests/month covers ~600 pages comfortably). No charge
should occur unless you exceed that quota.

This is a classic OCR engine (not an LLM) — generally solid for image
quality, but it doesn't understand context the way a multimodal model does.
For light novel pages with dense vertical prose, openrouter_ocr.py /
gemini_direct_ocr.py usually give better results with less setup friction.
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

from google.cloud import vision
from natsort import natsorted
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def ocr_image(client: vision.ImageAnnotatorClient, path: Path, retries: int = 3) -> str:
    """OCRs a single scan, returning text in reading order."""
    with io.open(path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    # The "ja" language hint helps the model handle vertical Japanese more accurately
    image_context = vision.ImageContext(language_hints=["ja"])

    last_err = None
    for attempt in range(retries):
        try:
            response = client.document_text_detection(
                image=image, image_context=image_context
            )
            if response.error.message:
                raise RuntimeError(response.error.message)
            return response.full_text_annotation.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to OCR {path.name} after {retries} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Batch-OCR a light novel via Google Cloud Vision")
    parser.add_argument("--input", required=True, help="Folder with scanned page images (jpg/png)")
    parser.add_argument("--output", required=True, help="Folder for the OCR results")
    parser.add_argument("--credentials", required=True, help="Path to the service-account key.json")
    parser.add_argument(
        "--start-page", type=int, default=1, help="Page number to start the header numbering from"
    )
    args = parser.parse_args()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.credentials

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

    client = vision.ImageAnnotatorClient()

    combined_path = output_dir / "combined.md"
    failed = []

    with open(combined_path, "w", encoding="utf-8") as combined_f:
        for idx, img_path in enumerate(tqdm(images, desc="OCR"), start=args.start_page):
            txt_out = pages_dir / f"{img_path.stem}.txt"

            # Skip pages already OCR'd — handy if a previous run was interrupted
            if txt_out.exists():
                text = txt_out.read_text(encoding="utf-8")
            else:
                try:
                    text = ocr_image(client, img_path)
                    txt_out.write_text(text, encoding="utf-8")
                except Exception as e:  # noqa: BLE001
                    print(f"\nError on {img_path.name}: {e}", file=sys.stderr)
                    failed.append(img_path.name)
                    text = ""
                    # Do NOT write a file to disk on failure — otherwise the next
                    # run would see the file exists and skip retrying it.

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
