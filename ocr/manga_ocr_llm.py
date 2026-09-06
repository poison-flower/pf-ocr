#!/usr/bin/env python3
"""
Batch-OCR (or batch-translate) for scanned manga pages via any
OpenAI-compatible API (OpenRouter, a direct provider endpoint, a
self-hosted proxy, etc.) — including Gemini, GPT-4V-class models, or
anything else exposed through such an endpoint.

Manga pages split their text across scattered speech bubbles, thought
bubbles, narration boxes, and SFX rather than running prose, so this script
has genuinely different logic from novel_ocr.py: it transcribes each page
as a numbered, typed, reading-order list of bubbles, and — since that's
what most people actually want out of a manga OCR pass — can translate
directly instead, with a glossary for consistent names/terms and a sliding
window of previous pages' translations for continuity.

Usage (after filling in config.json):
    python manga_ocr_llm.py --input ./pages --output ./out

    # translate instead of transcribing:
    python manga_ocr_llm.py --input ./pages --output ./out \\
        --translate --target-lang Russian

    # ...with a glossary of established name/term translations:
    python manga_ocr_llm.py --input ./pages --output ./out \\
        --translate --target-lang Russian --glossary ../ocr/glossary.md
    # (glossary.md sitting next to this script is picked up automatically
    # even without --glossary — see ocr/glossary.example.md for the format)

    # ...with more/less continuity context from previous pages (default: 2):
    python manga_ocr_llm.py --input ./pages --output ./out \\
        --translate --target-lang Russian --context-pages 4

    # ...with lookahead context from upcoming (untranslated) pages, in the
    # original Japanese — helps with twists, who's-talking-to-whom, jokes
    # that pay off a page later:
    python manga_ocr_llm.py --input ./pages --output ./out \\
        --translate --target-lang Russian --context-pages-ahead 1

Requirements:
    pip install openai pillow natsort tqdm

One-time setup:
    1. Copy config.example.json (repo root) -> config.json
    2. Fill in the "ocr" section: api_key, base_url, model (temperature/
       max_tokens/top_p/reasoning_effort are optional — see config.example.json)
    3. Optionally edit prompt_manga.txt (transcription) or
       prompt_manga_translate.txt (--translate) to fit your book / house style

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
from collections import deque
from datetime import datetime
from pathlib import Path

from natsort import natsorted
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))  # so `from local_mangaocr_ocr import ...` works when run from elsewhere


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


def load_prompt(prompt_path: Path, target_language: str = None, glossary: str = None) -> str:
    if not prompt_path.exists():
        print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if target_language:
        prompt = prompt.replace("{target_language}", target_language)
    if glossary:
        prompt += (
            "\n\n---\n\n"
            "Glossary: use these established translations for the following "
            "names/terms for consistency. Anything not listed here, translate "
            "naturally.\n\n" + glossary
        )
    return prompt


def find_glossary(explicit_path: str) -> Path:
    """Resolves the glossary path: explicit --glossary if given, otherwise
    an auto-detected glossary.md next to the ocr scripts themselves (a
    stable, book-independent location — handy for a future UI to edit)."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            print(f"Glossary file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path
    auto_path = SCRIPT_DIR / "glossary.md"
    return auto_path if auto_path.exists() else None


def build_context_block(past_pages: list, ahead_pages: list) -> str:
    """Builds a prompt suffix with:
    - past_pages: the last N *translated* pages, for continuity (character
      voice, an ongoing exchange, pronouns/referents that only make sense
      given what was just said).
    - ahead_pages: the next N pages' raw *Japanese* transcription (never a
      draft translation — see the ocr/README.md design note on why), for
      context that only becomes clear from what's about to happen (a
      twist, who's actually being addressed, a joke that pays off a page
      later).

    Both are lists of (page_name, text) tuples, oldest/nearest first.
    """
    if not past_pages and not ahead_pages:
        return ""
    parts = []
    if past_pages:
        parts.append(
            "\n\n---\n\n"
            "Context: translations of the immediately preceding page(s), for "
            "continuity only (character voice, an ongoing conversation, "
            "pronouns/referents). Do NOT re-translate or repeat any of this "
            "in your output — translate only the NEW page shown in the image.\n"
        )
        for name, text in past_pages:
            parts.append(f"\n[Previous page: {name}]\n{text}")
    if ahead_pages:
        parts.append(
            "\n\n---\n\n"
            "Context: the ORIGINAL JAPANESE (not a translation) of the "
            "upcoming page(s) that follow the one you're translating now. "
            "Use this only to correctly resolve things that depend on what "
            "happens next — pronoun/referent gender, who a line is actually "
            "addressed to, a setup whose payoff lands later. Do NOT "
            "translate or otherwise output any of this — translate only the "
            "CURRENT page shown in the image.\n"
        )
        for name, text in ahead_pages:
            parts.append(f"\n[Upcoming page ({name}), original Japanese]\n{text}")
    return "".join(parts)


def get_japanese_context(
    img_path: Path, context_src_dir: Path,
    ahead_ocr_backend: str, ahead_ocr_state: dict,
) -> str:
    """Resolves the raw Japanese transcription of a page used as lookahead
    context, in priority order:
      1. A .txt already sitting in context_src_dir with the same stem —
         hand-prepared, or cached from a previous run/page. Used as-is.
      2. Otherwise, OCR it now with the chosen backend (--ahead-ocr-backend)
         and cache the result into context_src_dir, so the next page that
         needs this same page as context doesn't re-OCR it.

    Returns "" (and prints a warning) if OCR isn't available/fails — a
    missing bit of lookahead context isn't worth failing the whole page
    over, it just means less context than requested.
    """
    cached_path = context_src_dir / f"{img_path.stem}.txt"
    if cached_path.exists():
        return cached_path.read_text(encoding="utf-8").strip()

    try:
        if ahead_ocr_backend == "local":
            if "mocr" not in ahead_ocr_state:
                from local_mangaocr_ocr import ocr_page_manga
                from manga_ocr import MangaOcr
                print("Loading local manga-ocr model for lookahead context (first use only)...")
                ahead_ocr_state["mocr"] = MangaOcr()
                ahead_ocr_state["ocr_page_manga"] = ocr_page_manga
            pil_img = Image.open(img_path)
            text = ahead_ocr_state["ocr_page_manga"](ahead_ocr_state["mocr"], pil_img)
        else:  # "llm"
            client = ahead_ocr_state["client"]
            model = ahead_ocr_state["model"]
            request_kwargs = ahead_ocr_state["request_kwargs"]
            extra_body = ahead_ocr_state["extra_body"]
            prompt = ahead_ocr_state["prompt"]
            text = ocr_image(client, model, prompt, img_path, request_kwargs, extra_body)
    except Exception as e:  # noqa: BLE001
        print(f"\nWarning: couldn't OCR {img_path.name} for lookahead context: {e}", file=sys.stderr)
        return ""

    text = text.strip()
    context_src_dir.mkdir(parents=True, exist_ok=True)
    cached_path.write_text(text, encoding="utf-8")
    return text


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
    parser = argparse.ArgumentParser(description="Batch-OCR/translate manga pages via an OpenAI-compatible API")
    parser.add_argument("--input", required=True, help="Folder with scanned page images (jpg/png)")
    parser.add_argument("--output", required=True, help="Folder for the OCR results")
    parser.add_argument(
        "--config", default=str(ROOT_DIR / "config.json"),
        help="Path to config.json with an \"ocr\" section (api_key/base_url/model/...). "
             "Defaults to config.json at the repo root."
    )
    parser.add_argument(
        "--translate", action="store_true",
        help="Translate the page instead of transcribing the Japanese "
             "(uses prompt_manga_translate.txt by default)"
    )
    parser.add_argument(
        "--target-lang", default="English",
        help="Target language for --translate, written out as a plain name "
             "(e.g. Russian, English, Spanish). Default: English"
    )
    parser.add_argument(
        "--glossary", default=None,
        help="[--translate only] Path to a glossary.md of established name/term "
             "translations, sent along with the prompt for consistency (see "
             "ocr/glossary.example.md). Auto-detected as glossary.md next to "
             "this script if not given."
    )
    parser.add_argument(
        "--context-pages", type=int, default=2,
        help="[--translate only] Include the last N translated pages in the "
             "prompt for continuity (character voice, ongoing dialogue, "
             "pronouns/referents). 0 disables it. Default: 2"
    )
    parser.add_argument(
        "--context-pages-ahead", type=int, default=0,
        help="[--translate only] Also include the ORIGINAL JAPANESE (never a "
             "draft translation — see ocr/README.md) of the next N pages, "
             "for context that depends on what happens next (a twist, "
             "who's actually being addressed, a joke that pays off later). "
             "0 (default) disables it. Before translating a page, this "
             "makes sure the next N pages' Japanese text is available "
             "(OCR'ing it now if there's no cached/hand-prepared .txt yet), "
             "so pages needing lookahead take a bit longer."
    )
    parser.add_argument(
        "--ahead-ocr-backend", choices=["llm", "local"], default="llm",
        help="[--context-pages-ahead only] How to OCR an upcoming page's "
             "Japanese text when no cached/hand-prepared .txt is found for "
             "it yet. 'llm': the same OpenAI-compatible API as the main "
             "translation (transcription prompt, no translation). 'local': "
             "manga-ocr running fully offline (requires the manga-ocr and "
             "opencv-python packages). Default: llm"
    )
    parser.add_argument(
        "--context-src-dir", default=None,
        help="[--context-pages-ahead only] Folder holding the upcoming pages' "
             "original-Japanese .txt files, named exactly like the page image "
             "stems (e.g. page005.txt for page005.jpg) — hand-prepared ones "
             "are used as-is; OCR'd-on-the-fly ones are cached here too. "
             "Defaults to <output>/context_src/"
    )
    parser.add_argument(
        "--prompt-file", default=None,
        help="Path to the prompt file (defaults to prompt_manga.txt, or "
             "prompt_manga_translate.txt with --translate)"
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

    if args.translate:
        default_prompt_name = "prompt_manga_translate.txt"
        target_language = args.target_lang
        glossary_path = find_glossary(args.glossary)
        glossary = glossary_path.read_text(encoding="utf-8").strip() if glossary_path else None
        if glossary_path:
            print(f"Using glossary: {glossary_path}")
    else:
        default_prompt_name = "prompt_manga.txt"
        target_language = None
        glossary = None
    prompt_file = Path(args.prompt_file) if args.prompt_file else SCRIPT_DIR / default_prompt_name
    prompt = load_prompt(prompt_file, target_language, glossary)

    use_ahead_context = args.translate and args.context_pages_ahead > 0
    ahead_ocr_state = {}
    if use_ahead_context and args.ahead_ocr_backend == "llm":
        # Plain transcription prompt/params for OCR'ing lookahead pages —
        # never the translate prompt (we want the raw Japanese, not a
        # draft translation — see the design note in build_context_block).
        ahead_ocr_state["prompt"] = load_prompt(SCRIPT_DIR / "prompt_manga.txt")
        ahead_ocr_state["request_kwargs"] = request_kwargs
        ahead_ocr_state["extra_body"] = extra_body
        # client/model are filled in below, once the real client exists.

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

    use_context = args.translate and args.context_pages > 0
    recent_context = deque(maxlen=args.context_pages) if use_context else None

    context_src_dir = Path(args.context_src_dir) if args.context_src_dir else output_dir / "context_src"

    client = OpenAI(base_url=base_url, api_key=api_key)

    if use_ahead_context and args.ahead_ocr_backend == "llm":
        ahead_ocr_state["client"] = client
        ahead_ocr_state["model"] = model

    combined_path = output_dir / "combined.md"
    failed = []

    with open(combined_path, "w", encoding="utf-8") as combined_f:
        for pos, img_path in enumerate(tqdm(images, desc="OCR")):
            idx = args.start_page + pos
            txt_out = pages_dir / f"{img_path.stem}.txt"

            if txt_out.exists():
                text = txt_out.read_text(encoding="utf-8")
            else:
                page_prompt = prompt
                context_used_past = []
                context_used_ahead = []
                if use_context and recent_context:
                    context_used_past = list(recent_context)
                if use_ahead_context:
                    for future_img in images[pos + 1: pos + 1 + args.context_pages_ahead]:
                        jp_text = get_japanese_context(future_img, context_src_dir, args.ahead_ocr_backend, ahead_ocr_state)
                        if jp_text and jp_text != "[NO_TEXT]":
                            context_used_ahead.append((future_img.stem, jp_text))
                if context_used_past or context_used_ahead:
                    page_prompt = prompt + build_context_block(context_used_past, context_used_ahead)

                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "backend": "manga_ocr_llm",
                    "page": img_path.name,
                    "translate": args.translate,
                    "target_lang": args.target_lang if args.translate else None,
                    "glossary_used": bool(glossary),
                    "context_pages_used": [name for name, _ in context_used_past],
                    "context_pages_ahead_used": [name for name, _ in context_used_ahead],
                    "model": model,
                    "base_url": base_url,
                } if log_dir is not None else None
                try:
                    text = ocr_image(client, model, page_prompt, img_path, request_kwargs, extra_body, log_entry=log_entry)
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

            # Feed this page's (already-completed-or-just-translated) text
            # forward as context for the following pages. Blank/[NO_TEXT]
            # pages carry no useful continuity, so skip adding those.
            if use_context and text.strip() and text.strip() != "[NO_TEXT]":
                recent_context.append((img_path.stem, text.strip()))

            combined_f.write(f"\n\n<!-- page {idx}: {img_path.name} -->\n\n")
            combined_f.write(text)

    print(f"\nDone. Combined file: {combined_path}")
    print(f"Per-page files: {pages_dir}")
    if use_ahead_context:
        print(f"Cached lookahead Japanese context: {context_src_dir}")
    if log_dir is not None:
        print(f"Request/response logs: {log_dir}")
    if failed:
        print(f"\nFailed to OCR {len(failed)} page(s):")
        for name in failed:
            print(f"  - {name}")
        print("Re-run the script with the same --output folder — already-done pages will not be redone.")


if __name__ == "__main__":
    main()
