#!/usr/bin/env python3
"""
Fully local, offline batch-OCR for scanned pages of a Japanese light novel
(vertical text) via manga-ocr — no cloud account or credit card required.

Usage:
    python local_mangaocr_ocr.py --input ./pages --output ./out

Requirements:
    pip install manga-ocr opencv-python pillow natsort tqdm

The first run downloads manga-ocr's model weights (~400 MB) from
HuggingFace; after that everything runs offline. Without a GPU it runs on
CPU, just slower (roughly 1-3 sec per column).

How it works:
    1. Each page is cut into vertical text columns (by detecting whitespace
       gaps between columns — typical light novel layout).
    2. Columns are sorted right-to-left (the reading order for vertical
       Japanese text).
    3. Each column is OCR'd separately via manga-ocr.
    4. Results are joined back into per-page text.

manga-ocr was trained mainly on manga speech bubbles (short text blocks),
not dense full-page prose, so column segmentation matters a lot here for
quality. If segmentation performs poorly on your scans (e.g. unusual
layout), pass --whole-page to feed the model the full page without cutting
it into columns (simpler, but usually lower quality on dense prose).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def find_columns(img_gray: np.ndarray, min_col_width: int = 12, gap_threshold: int = 4):
    """Finds x-ranges of vertical text columns via a pixel-density projection.

    Returns a list of (x_start, x_end), sorted RIGHT-TO-LEFT (the reading
    order for vertical Japanese text).
    """
    # Binarize: text (dark) -> white, background -> black
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate vertically a bit to merge characters within a column into one solid strip
    kernel = np.ones((25, 1), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    col_sums = dilated.sum(axis=0)  # text density per pixel column
    has_text = col_sums > 0

    columns = []
    x = 0
    width = len(has_text)
    while x < width:
        if has_text[x]:
            start = x
            while x < width and (has_text[x] or _gap_too_small(has_text, x, gap_threshold)):
                x += 1
            end = x
            if end - start >= min_col_width:
                columns.append((start, end))
        else:
            x += 1

    columns.sort(key=lambda c: c[0], reverse=True)  # right-to-left
    return columns


def _gap_too_small(has_text: np.ndarray, x: int, gap_threshold: int) -> bool:
    """Checks whether a text-free gap is shorter than gap_threshold (to avoid splitting a column needlessly)."""
    if has_text[x]:
        return False
    end = x
    while end < len(has_text) and not has_text[end]:
        end += 1
    return (end - x) < gap_threshold


def ocr_page(mocr, pil_img: Image.Image, whole_page: bool) -> str:
    if whole_page:
        return mocr(pil_img)

    img_np = np.array(pil_img.convert("L"))
    columns = find_columns(img_np)

    if not columns:
        # No columns detected (e.g. an illustration-only page) — fall back to the whole page
        return mocr(pil_img)

    texts = []
    for x_start, x_end in columns:
        pad = 4
        crop = pil_img.crop((max(0, x_start - pad), 0, min(pil_img.width, x_end + pad), pil_img.height))
        text = mocr(crop)
        if text.strip():
            texts.append(text.strip())

    return "\n".join(texts)


def main():
    parser = argparse.ArgumentParser(description="Local batch-OCR for a light novel via manga-ocr")
    parser.add_argument("--input", required=True, help="Folder with scanned page images (jpg/png)")
    parser.add_argument("--output", required=True, help="Folder for the OCR results")
    parser.add_argument(
        "--whole-page", action="store_true",
        help="Skip column segmentation, feed the whole page to the model at once"
    )
    parser.add_argument("--start-page", type=int, default=1)
    args = parser.parse_args()

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
    print("Loading the manga-ocr model (downloads weights on first run, ~400 MB)...")

    from manga_ocr import MangaOcr
    mocr = MangaOcr()

    combined_path = output_dir / "combined.md"
    failed = []

    with open(combined_path, "w", encoding="utf-8") as combined_f:
        for idx, img_path in enumerate(tqdm(images, desc="OCR"), start=args.start_page):
            txt_out = pages_dir / f"{img_path.stem}.txt"

            if txt_out.exists():
                text = txt_out.read_text(encoding="utf-8")
            else:
                try:
                    pil_img = Image.open(img_path)
                    text = ocr_page(mocr, pil_img, args.whole_page)
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
