#!/usr/bin/env python3
"""
Batch-OCR for scanned pages of a Japanese light novel (vertical text) via any
OpenAI-compatible API (OpenRouter, a direct provider endpoint, a
self-hosted proxy, etc.) — including Gemini, GPT-4V-class models, or
anything else exposed through such an endpoint.

This lets a multimodal LLM "read" a page image directly, including
vertical Japanese text, without any column-segmentation preprocessing.

Deliberately simple: light novel pages are dense running prose, and all
this script needs to do is transcribe them accurately. It has no
translate/glossary/context-page machinery — see manga_ocr_llm.py for that,
which is a genuinely different job (short, scattered dialogue that needs
continuity handling).

Usage (after filling in config.json):
    python novel_ocr.py --input ./pages --output ./out

Requirements:
    pip install openai pillow natsort tqdm

One-time setup:
    1. Copy config.example.json (repo root) -> config.json
    2. Fill in the "ocr" section: api_key, base_url, model (temperature/
       max_tokens/top_p/reasoning_effort are optional — see config.example.json)
    3. Optionally edit prompt_novel.txt to fit your book / house style

Every request/response is logged as one JSON file under logs/ at the repo
root (--log-dir to change, --no-log to disable) — full prompt, model
params, every retry attempt, and the full raw API response (the base64
image itself is never included, only its path/size).
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from natsort import natsorted
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def load_config(config_path: Path) -> dict:
    """Loads the shared config.json and returns its "ocr" section.

    Falls back to treating the whole file as the ocr config if there's no
    "ocr" key, so a bare {"api_key": ...} style file still works.
    """
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Failed to parse {config_path}: {e}", file=sys.stderr)
        sys.exit(1)
    return data.get("ocr", data) if isinstance(data, dict) else {}


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    return prompt_path.read_text(encoding="utf-8").strip()


def build_request_kwargs(config: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    """Resolves OpenAI-API-style request params: CLI flag > config.json > a
    sensible default. Returns (kwargs, extra_body) — standard params go
    straight into the request; reasoning_effort goes through extra_body
    since support for it varies by provider/model and extra_body is the
    designed passthrough for exactly that.
    """
    temperature = args.temperature if args.temperature is not None else config.get("temperature", 0)
    max_tokens = args.max_tokens if args.max_tokens is not None else config.get("max_tokens")
    top_p = args.top_p if args.top_p is not None else config.get("top_p")
    reasoning_effort = args.reasoning_effort or config.get("reasoning_effort")

    kwargs = {"temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if top_p is not None:
        kwargs["top_p"] = top_p

    extra_body = {}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort

    return kwargs, extra_body


def write_log(log_dir: Path, page_name: str, entry: dict) -> None:
    """Writes one JSON log file per request/response.

    The base64 image payload itself is never included (megabytes of no
    debugging value, one per page) — only its path/size — but everything
    else (full prompt text, model params, every retry attempt, the full raw
    API response) is recorded as-is.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", page_name)
    log_path = log_dir / f"{timestamp}_{safe_name}.json"
    log_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


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


def ocr_image(
    client: OpenAI, model: str, prompt: str, path: Path,
    request_kwargs: dict, extra_body: dict,
    retries: int = 3, log_entry: dict = None,
) -> str:
    data_url = image_to_data_url(path)

    if log_entry is not None:
        log_entry["request"] = {
            "model": model,
            **request_kwargs,
            "extra_body": extra_body or None,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"<omitted: {path.name}, {path.stat().st_size} bytes>"}},
                    ],
                }
            ],
        }
        log_entry["attempts"] = []

    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                extra_body=extra_body or None,
                **request_kwargs,
            )
            text = (response.choices[0].message.content or "").strip()
            if log_entry is not None:
                try:
                    raw_response = response.model_dump()
                except Exception:  # noqa: BLE001
                    raw_response = {"content": text}
                log_entry["attempts"].append({"attempt": attempt + 1, "success": True, "response": raw_response})
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            if log_entry is not None:
                log_entry["attempts"].append({"attempt": attempt + 1, "success": False, "error": str(e)})
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to OCR {path.name} after {retries} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Batch-OCR a light novel via an OpenAI-compatible API")
    parser.add_argument("--input", required=True, help="Folder with scanned page images (jpg/png)")
    parser.add_argument("--output", required=True, help="Folder for the OCR results")
    parser.add_argument(
        "--config", default=str(ROOT_DIR / "config.json"),
        help="Path to config.json with an \"ocr\" section (api_key/base_url/model/...). "
             "Defaults to config.json at the repo root."
    )
    parser.add_argument(
        "--prompt-file", default=str(SCRIPT_DIR / "prompt_novel.txt"),
        help="Path to the prompt file (defaults to prompt_novel.txt next to this script)"
    )
    parser.add_argument("--base-url", default=None, help="Override base_url from config.json")
    parser.add_argument("--model", default=None, help="Override model from config.json")
    parser.add_argument("--api-key", default=None, help="Override api_key from config.json")
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override temperature from config.json (default if unset anywhere: 0)"
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_tokens from config.json")
    parser.add_argument("--top-p", type=float, default=None, help="Override top_p from config.json")
    parser.add_argument(
        "--reasoning-effort", default=None,
        help="Override reasoning_effort from config.json (e.g. low/medium/high — support "
             "depends on the model/provider; omitted from the request unless set)"
    )
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="Delay in seconds between requests (useful if you're hitting rate limits)"
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

    request_kwargs, extra_body = build_request_kwargs(config, args)
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

    log_dir = None if args.no_log else Path(args.log_dir)

    client = OpenAI(base_url=base_url, api_key=api_key)

    combined_path = output_dir / "combined.md"
    failed = []

    with open(combined_path, "w", encoding="utf-8") as combined_f:
        for idx, img_path in enumerate(tqdm(images, desc="OCR"), start=args.start_page):
            txt_out = pages_dir / f"{img_path.stem}.txt"

            if txt_out.exists():
                text = txt_out.read_text(encoding="utf-8")
            else:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "backend": "novel_ocr",
                    "page": img_path.name,
                    "model": model,
                    "base_url": base_url,
                } if log_dir is not None else None
                try:
                    text = ocr_image(client, model, prompt, img_path, request_kwargs, extra_body, log_entry=log_entry)
                    txt_out.write_text(text, encoding="utf-8")
                except Exception as e:  # noqa: BLE001
                    print(f"\nError on {img_path.name}: {e}", file=sys.stderr)
                    failed.append(img_path.name)
                    text = ""
                    if log_entry is not None:
                        log_entry["error"] = str(e)
                    # Do NOT write a file to disk on failure — otherwise the next
                    # run would see the file exists and assume the page is already
                    # done, silently skipping a retry forever.
                if log_entry is not None:
                    write_log(log_dir, img_path.stem, log_entry)
                if args.sleep:
                    time.sleep(args.sleep)

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
