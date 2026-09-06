#!/usr/bin/env python3
"""
Batch-OCR for scanned pages of a Japanese light novel (vertical text) via
Google Cloud Vision.

Usage:
    python google_vision_ocr.py --input ./pages --output ./out
    # (credentials path comes from config.json's vision.credentials, or pass --credentials)

    # for manga instead of a light novel:
    python google_vision_ocr.py --input ./pages --output ./out --mode manga

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
quality, but it doesn't understand context the way a multimodal model does,
and (unlike the LLM-based scripts) it can't translate. For light novel
pages with dense vertical prose, novel_ocr.py usually gives better results
with less setup friction; for manga, see manga_ocr_llm.py.

Every request/response is logged as one JSON file under logs/ at the repo
root (--log-dir to change, --no-log to disable).
"""

import argparse
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from google.cloud import vision
from natsort import natsorted
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def load_config(config_path: Path) -> dict:
    """Loads the shared config.json and returns its "vision" section
    (Google Cloud Vision has nothing to do with the OpenAI-compatible "ocr"
    section used by novel_ocr.py / manga_ocr_llm.py)."""
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Failed to parse {config_path}: {e}", file=sys.stderr)
        sys.exit(1)
    return data.get("vision", data) if isinstance(data, dict) else {}


def write_log(log_dir: Path, page_name: str, entry: dict) -> None:
    """Writes one JSON log file per request/response (see novel_ocr.py
    for the rationale — same format, minus the image payload itself)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", page_name)
    log_path = log_dir / f"{timestamp}_{safe_name}.json"
    log_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _response_to_dict(response) -> dict:
    try:
        return vision.AnnotateImageResponse.to_dict(response)
    except Exception:  # noqa: BLE001
        return {"full_text_annotation_text": getattr(getattr(response, "full_text_annotation", None), "text", None)}


def ocr_image_novel(client: vision.ImageAnnotatorClient, path: Path, retries: int = 3, log_entry: dict = None) -> str:
    """OCRs a single scan of running prose, returning text in reading order."""
    with io.open(path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    # The "ja" language hint helps the model handle vertical Japanese more accurately
    image_context = vision.ImageContext(language_hints=["ja"])

    if log_entry is not None:
        log_entry["request"] = {
            "feature": "document_text_detection",
            "language_hints": ["ja"],
            "image": f"<omitted: {path.name}, {path.stat().st_size} bytes>",
        }
        log_entry["attempts"] = []

    last_err = None
    for attempt in range(retries):
        try:
            response = client.document_text_detection(
                image=image, image_context=image_context
            )
            if response.error.message:
                raise RuntimeError(response.error.message)
            text = response.full_text_annotation.text
            if log_entry is not None:
                log_entry["attempts"].append(
                    {"attempt": attempt + 1, "success": True, "response": _response_to_dict(response)}
                )
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            if log_entry is not None:
                log_entry["attempts"].append({"attempt": attempt + 1, "success": False, "error": str(e)})
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to OCR {path.name} after {retries} attempts: {last_err}")


def ocr_image_manga(client: vision.ImageAnnotatorClient, path: Path, retries: int = 3, log_entry: dict = None) -> str:
    """OCRs a single manga page.

    document_text_detection assumes a running paragraph flow, which falls
    apart on manga: bubbles are scattered blocks, not one paragraph. Instead
    this groups Vision's per-paragraph bounding boxes into text-block
    "clusters" and orders them in manga reading order: clusters right-to-left
    by their rightmost edge, breaking ties top-to-bottom.
    """
    with io.open(path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=["ja"])

    if log_entry is not None:
        log_entry["request"] = {
            "feature": "document_text_detection",
            "language_hints": ["ja"],
            "image": f"<omitted: {path.name}, {path.stat().st_size} bytes>",
        }
        log_entry["attempts"] = []

    last_err = None
    for attempt in range(retries):
        try:
            response = client.document_text_detection(
                image=image, image_context=image_context
            )
            if response.error.message:
                raise RuntimeError(response.error.message)
            if log_entry is not None:
                log_entry["attempts"].append(
                    {"attempt": attempt + 1, "success": True, "response": _response_to_dict(response)}
                )
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if log_entry is not None:
                log_entry["attempts"].append({"attempt": attempt + 1, "success": False, "error": str(e)})
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to OCR {path.name} after {retries} attempts: {last_err}")

    blocks = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            xs = [v.x for v in block.bounding_box.vertices]
            ys = [v.y for v in block.bounding_box.vertices]
            text = ""
            for paragraph in block.paragraphs:
                words = []
                for word in paragraph.words:
                    words.append("".join(s.text for s in word.symbols))
                text += "".join(words)
            if text.strip():
                blocks.append({"text": text.strip(), "x_max": max(xs), "y_min": min(ys)})

    if not blocks:
        result = "[NO_TEXT]"
    else:
        # Manga reading order: right-to-left, breaking ties top-to-bottom. This
        # is a coarse heuristic (true panel/bubble order can't be recovered from
        # plain bounding boxes) — always spot-check against the page.
        blocks.sort(key=lambda b: (-b["x_max"], b["y_min"]))
        result = "\n".join(f"{i}. {b['text']}" for i, b in enumerate(blocks, start=1))

    if log_entry is not None:
        log_entry["reading_order_result"] = result
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch-OCR a light novel via Google Cloud Vision")
    parser.add_argument("--input", required=True, help="Folder with scanned page images (jpg/png)")
    parser.add_argument("--output", required=True, help="Folder for the OCR results")
    parser.add_argument(
        "--credentials", default=None,
        help="Path to the service-account key.json (defaults to config.json's vision.credentials)"
    )
    parser.add_argument(
        "--config", default=str(ROOT_DIR / "config.json"),
        help="Path to config.json with a \"vision\" section. Defaults to config.json at the repo root."
    )
    parser.add_argument(
        "--mode", choices=["novel", "manga"], default="novel",
        help="novel: dense running prose, full-page reading order (default). "
             "manga: scattered bubbles, grouped and sorted in manga reading order."
    )
    parser.add_argument(
        "--start-page", type=int, default=1, help="Page number to start the header numbering from"
    )
    parser.add_argument(
        "--log-dir", default=str(ROOT_DIR / "logs"),
        help="Folder for per-page request/response logs (one JSON file per page). "
             "Defaults to logs/ at the repo root."
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Disable request/response logging entirely"
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    credentials = args.credentials or config.get("credentials")
    if not credentials:
        print(
            f"No credentials found. Set vision.credentials in {args.config} or pass --credentials.",
            file=sys.stderr,
        )
        sys.exit(1)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

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

    log_dir = None if args.no_log else Path(args.log_dir)

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
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "backend": "google_vision",
                    "page": img_path.name,
                    "mode": args.mode,
                } if log_dir is not None else None
                try:
                    ocr_fn = ocr_image_manga if args.mode == "manga" else ocr_image_novel
                    text = ocr_fn(client, img_path, log_entry=log_entry)
                    txt_out.write_text(text, encoding="utf-8")
                except Exception as e:  # noqa: BLE001
                    print(f"\nError on {img_path.name}: {e}", file=sys.stderr)
                    failed.append(img_path.name)
                    text = ""
                    if log_entry is not None:
                        log_entry["error"] = str(e)
                    # Do NOT write a file to disk on failure — otherwise the next
                    # run would see the file exists and skip retrying it.
                if log_entry is not None:
                    write_log(log_dir, img_path.stem, log_entry)

            combined_f.write(f"\n\n<!-- page {idx}: {img_path.name} -->\n\n")
            combined_f.write(text)

    print(f"\nDone. Combined file: {combined_path}")
    print(f"Per-page files: {pages_dir}")
    if log_dir is not None:
        print(f"Request/response logs: {log_dir}")
    if failed:
        print(f"\nFailed to OCR {len(failed)} page(s):")
        for name in failed:
            print(f"  - {name}")
        print("Re-run the script with the same --output folder — already-done pages will not be redone.")


if __name__ == "__main__":
    main()
