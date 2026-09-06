#!/usr/bin/env python3
"""
Fully local, offline batch-OCR for scanned Japanese pages via manga-ocr —
no cloud account or credit card required. Supports two page layouts via
--mode:

    novel (default) — dense running prose (light novel pages). Each page is
        cut into vertical text columns (by detecting whitespace gaps between
        columns), columns are sorted right-to-left, and each is OCR'd
        separately, since manga-ocr expects short text blocks rather than a
        whole page of prose.

    manga — speech bubbles / narration boxes scattered over one or more
        panels. Bubbles are detected via their outline shape (a closed,
        fairly convex blob of ink enclosing a lighter fill), sorted into an
        approximate manga reading order (right-to-left, top-to-bottom, with
        panel rows inferred from vertical overlap), and each is OCR'd
        separately as a whole bubble crop — the input format manga-ocr was
        actually trained on.

Usage:
    python local_mangaocr_ocr.py --input ./pages --output ./out
    python local_mangaocr_ocr.py --input ./pages --output ./out --mode manga

Requirements:
    pip install manga-ocr opencv-python pillow natsort tqdm

The first run downloads manga-ocr's model weights (~400 MB) from
HuggingFace; after that everything runs offline. Without a GPU it runs on
CPU, just slower (roughly 1-3 sec per column/bubble).

Notes on --mode manga:
    - Bubble detection is a geometric heuristic (contour shape + fill), not
      a trained detector, so it can miss borderless bubbles, split a bubble
      with a long speaker "tail", or get the reading order wrong on unusual
      layouts. Pass --debug to also write an annotated copy of each page
      (numbered boxes) to <output>/bubble_debug/ so you can quickly spot
      and manually fix any misordered or missed pages in the .txt output.
    - A page with no detected bubbles (splash art, etc.) gets [NO_TEXT].
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


def find_bubbles(img_gray: np.ndarray, min_area_frac: float = 0.0015, max_area_frac: float = 0.35):
    """Detects speech-bubble-like shapes and returns their bounding boxes.

    Speech bubbles are (usually) a closed ink outline enclosing a lighter
    fill. Binarizing+inverting turns that outline into a blob whose *outer*
    contour is a good stand-in for the bubble's overall shape, so bubbles
    can be picked out by area and convexity without needing a trained
    detector:
        1. Otsu-threshold + invert: ink -> white, everything else -> black.
        2. Morphological close: bridges small gaps in the outline (dashed
           bubble borders, a bubble "tail", anti-aliasing) so it forms one
           solid ring instead of several fragments.
        3. External contours only (RETR_EXTERNAL): a bubble's ring becomes one
           blob-like contour; panel frames and page borders are filtered out
           by area/aspect below.
        4. Keep contours that are a plausible bubble: not too small/large
           relative to the page, and fairly convex (area close to its
           convex-hull area) — panel borders, gutters, and stray ink specks
           don't pass this.

    Returns a list of (x, y, w, h) bounding boxes, unsorted, deduplicated.
    """
    page_area = img_gray.shape[0] * img_gray.shape[1]

    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # RETR_LIST (not RETR_EXTERNAL): a bubble drawn inside a panel is nested
    # inside the panel border's contour, so RETR_EXTERNAL would only return
    # the panel border and miss every bubble in it. RETR_LIST returns every
    # contour (panel borders, bubble rings, stray marks); the area/aspect/
    # solidity filters below do the actual selecting.
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_frac * page_area or area > max_area_frac * page_area:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < 0.55:  # panel borders / stray ink are far less convex than a bubble
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h else 0
        if aspect < 0.15 or aspect > 6.0:  # rule out thin frame edges/gutter slivers
            continue

        candidates.append((area, (x, y, w, h)))

    # A bubble's outline has thickness, so its outer and inner edge each
    # produce their own (near-identical, nested) contour — keep only the
    # larger of each such pair via simple greedy IoU suppression.
    candidates.sort(key=lambda c: c[0], reverse=True)
    boxes = []
    for _, box in candidates:
        if not any(_iou(box, kept) > 0.5 for kept in boxes):
            boxes.append(box)

    return boxes


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


def sort_manga_reading_order(boxes, row_overlap_ratio: float = 0.4):
    """Sorts bounding boxes into an approximate manga reading order.

    Groups boxes into "rows" (panel bands) by vertical overlap, orders rows
    top-to-bottom, then orders boxes within a row right-to-left — the
    standard reading order for a Japanese manga page. This is a heuristic:
    layouts with tall panels spanning multiple "rows" of a neighboring
    column can still come out wrong, which is what --debug is for.
    """
    remaining = sorted(boxes, key=lambda b: b[1])  # top-to-bottom as a starting point
    rows = []
    for box in remaining:
        x, y, w, h = box
        placed = False
        for row in rows:
            ry_min = min(b[1] for b in row)
            ry_max = max(b[1] + b[3] for b in row)
            overlap = min(y + h, ry_max) - max(y, ry_min)
            if overlap > row_overlap_ratio * min(h, ry_max - ry_min):
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    rows.sort(key=lambda row: min(b[1] for b in row))
    ordered = []
    for row in rows:
        row.sort(key=lambda b: b[0] + b[2], reverse=True)  # right edge, right-to-left
        ordered.extend(row)
    return ordered


def ocr_page_manga(mocr, pil_img: Image.Image, debug_path: Path = None) -> str:
    img_np = np.array(pil_img.convert("L"))
    boxes = find_bubbles(img_np)

    if not boxes:
        return "[NO_TEXT]"

    ordered = sort_manga_reading_order(boxes)

    if debug_path is not None:
        debug_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(ordered, start=1):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(debug_img, str(i), (x + 4, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_img)

    entries = []
    for i, (x, y, w, h) in enumerate(ordered, start=1):
        pad = 4
        crop = pil_img.crop((
            max(0, x - pad), max(0, y - pad),
            min(pil_img.width, x + w + pad), min(pil_img.height, y + h + pad),
        ))
        text = mocr(crop)
        if text.strip():
            entries.append(f"{i}. {text.strip()}")

    return "\n".join(entries) if entries else "[NO_TEXT]"


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
        "--mode", choices=["novel", "manga"], default="novel",
        help="novel: column-segmented running prose (default). "
             "manga: bubble-detected panels, sorted in manga reading order."
    )
    parser.add_argument(
        "--whole-page", action="store_true",
        help="[novel mode] Skip column segmentation, feed the whole page to the model at once"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="[manga mode] Also save annotated pages (numbered bubble boxes) to "
             "<output>/bubble_debug/, to sanity-check detection/reading order"
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
                    if args.mode == "manga":
                        debug_path = (
                            output_dir / "bubble_debug" / f"{img_path.stem}.jpg"
                            if args.debug else None
                        )
                        text = ocr_page_manga(mocr, pil_img, debug_path)
                    else:
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
