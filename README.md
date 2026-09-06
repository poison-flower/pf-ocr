# Scanned Light Novel / Manga -> EPUB & Translation Tooling

Turns a folder of scanned Japanese pages into either a proper `.epub`
(light novels) or transcribed/translated text (manga), using a multimodal
LLM to do the OCR instead of a traditional column-segmentation pipeline.
Light novel and manga pages get separate tooling, since their layouts need
genuinely different logic — dense running prose vs. scattered speech
bubbles that benefit from translation, a glossary, and cross-page context.

## Pipeline

```
scanned page images
        │
        ▼
   ocr/ module          → transcribes (or translates) each page into
                            pages_txt/<name>.txt
        │
        ▼
  (manual step,          → sort the .txt files (and any illustration
   light novel only)       images) into chapters/chNN_name/ folders
        │
        ▼
epub_builder/ module     → assembles chapters/ into a finished .epub
                            (light novel only — manga output is meant for
                            your own typesetting workflow instead)
```

1. **[`ocr/`](ocr/README.md)** — batch-OCRs scanned pages into per-page
   `.txt` files, or translates them directly. `novel_ocr.py` (light novel)
   and `manga_ocr_llm.py` (manga) are the recommended entry points, both
   via any OpenAI-compatible API; `google_vision_ocr.py` (Google Cloud
   Vision) and `local_mangaocr_ocr.py` (fully offline `manga-ocr`) are
   pure-OCR alternatives with no translation.
2. **Manual sorting** (light novel only) — split the resulting pages into
   chapter folders (`ch00_frontmatter/`, `ch01_chapter00/`, ...), optionally
   dropping in a `cover.jpg` and illustration images alongside the `.txt`
   files. This step is manual because automatically detecting chapter
   boundaries from OCR'd headers turned out to be unreliable — folder
   structure is simple and unambiguous instead.
3. **[`epub_builder/`](epub_builder/README.md)** — assembles the sorted
   `chapters/` folder into a valid `.epub`: furigana notation becomes
   `<ruby>` markup, images are placed inline, a font from `font/` gets
   embedded, and metadata comes from the root `config.json`.

Both modules read their settings from a single `config.json` at the repo
root (copy `config.example.json` to get started) — see each module's own
README for the exact fields.

See each module's own README for setup and usage details.

## Quick start

```bash
# 0. One-time setup: copy the shared config and fill it in
cp config.example.json config.json
# fill in ocr.api_key / ocr.base_url / ocr.model, epub.title / epub.author / ...

# 1a. OCR a light novel
cd ocr
pip install openai pillow natsort tqdm
python novel_ocr.py --input /path/to/scans --output ./out

# 1b. ...or OCR/translate manga instead
python manga_ocr_llm.py --input /path/to/scans --output ./out
python manga_ocr_llm.py --input /path/to/scans --output ./out --translate --target-lang Russian

# 2. (light novel) Sort ./out/pages_txt/*.txt by hand into epub_builder/chapters/chNN_name/

# 3. (light novel) Build the epub
cd ../epub_builder
pip install natsort
python build_epub.py
```

## ⚠️ A note on copyright

This repository contains only the **tooling**. It is not meant to, and
should not, be used to host or distribute:
- scanned page images,
- OCR'd or translated text extracted from a copyrighted book,
- or a resulting `.epub` file,

for any book you don't hold the rights to. The `.gitignore` in this repo
already excludes `chapters/`, `pages_txt/`, `out/`, and `*.epub` for this
reason — keep it that way if you fork or extend this project. This tool is
intended for personal-use digitization of books you own, not redistribution.

## Requirements

See [`requirements.txt`](requirements.txt) for the full list. Not every
dependency is needed at once — install only what the OCR backend and
features you're using require (see each module's README).

## Status

This is a personal toolkit, still evolving. Contributions/forks welcome,
but expect rough edges — issues and PRs are handled best-effort.
