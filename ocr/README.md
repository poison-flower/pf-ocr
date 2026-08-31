# OCR module

Batch-transcribes scanned light novel pages (vertical Japanese text) into
per-page `.txt` files. This is step 1 of the pipeline — step 2 is the
[`epub_builder`](../epub_builder/README.md) module, which turns these `.txt`
files into a finished `.epub`.

## Choosing a backend

| Script | Backend | Setup needed | Notes |
|---|---|---|---|
| `openrouter_ocr.py` | Multimodal LLM via any OpenAI-compatible API | API key + base URL | **Recommended.** Reads a whole page at once, no column segmentation needed. |
| `gemini_direct_ocr.py` | Gemini API directly | Google AI Studio API key | Same idea as above, direct instead of via a proxy. |
| `google_vision_ocr.py` | Google Cloud Vision (classic OCR) | Google Cloud project + billing enabled | No LLM context understanding, but solid on clean scans. Free tier covers a typical light novel volume, but Google still requires a billing account to be linked. |
| `local_mangaocr_ocr.py` | [manga-ocr](https://github.com/kha-white/manga-ocr), fully offline | None — no account, no API key | Slower to set up quality-wise: manga-ocr expects short text blocks, so this script auto-segments each page into vertical columns before OCR-ing each one. |

If you have API access to a multimodal model (Gemini, GPT-4V-class models,
etc.) through any provider, `openrouter_ocr.py` is the easiest and generally
gives the best results with the least fuss.

## Setup for `openrouter_ocr.py` (recommended)

```bash
pip install openai pillow natsort tqdm
```

1. Copy `config.example.json` to `config.json` and fill in your `api_key`,
   `base_url` (your provider's OpenAI-compatible endpoint, usually ending in
   `/v1`), and `model` identifier.
2. Optionally edit `prompt.txt` — it's plain English text, no need to touch
   any code to tweak the instructions given to the model.

```bash
python openrouter_ocr.py --input ./pages --output ./out
```

- `--input`: folder with scanned page images (jpg/png/...), named so that
  alphabetical sorting matches page order (natural sort is used, so
  `page2.jpg` and `page10.jpg` sort correctly too).
- `--output`: folder for results. Creates `pages_txt/<name>.txt` (one file
  per page) plus a `combined.md` preview of the whole run.
- If interrupted, just re-run with the same `--output` — pages that already
  have a `.txt` file are skipped, so nothing already done gets re-sent
  (and re-billed).
- A page that fails to OCR (network error, rate limit, etc.) does **not**
  get an empty file written for it, specifically so the next run retries it
  instead of silently treating it as done.
- A page the model judges to be empty (illustration-only, blank, or a
  cover/technical page) gets a file containing exactly the literal text
  `[NO_TEXT]` — this is a deliberate marker, not an OCR failure. The
  `epub_builder` module knows to treat it as "no text on this page".
- `--sleep N` adds a delay (seconds) between requests if you're hitting
  rate limits.

## Other backends

`gemini_direct_ocr.py`, `google_vision_ocr.py`, and `local_mangaocr_ocr.py`
follow the same `--input`/`--output` convention and produce the same
`pages_txt/*.txt` + `combined.md` output — see each script's own docstring
for backend-specific setup.

## Next step

Once you have a `pages_txt/` folder full of `.txt` files, manually sort the
pages into the folder structure `epub_builder` expects (see
[`epub_builder/README.md`](../epub_builder/README.md)), then run the epub
builder.
