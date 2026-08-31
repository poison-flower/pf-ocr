# Light Novel Scan -> EPUB

Turns a folder of scanned light novel page images (vertical Japanese text)
into a proper `.epub`, with furigana rendered as real `<ruby>` markup,
illustrations kept in place, an embedded font, and epub metadata — using a
multimodal LLM to do the OCR instead of a traditional column-segmentation
pipeline.

## Pipeline

```
scanned page images
        │
        ▼
   ocr/ module          → transcribes each page into pages_txt/<name>.txt
        │
        ▼
  (manual step)          → sort the .txt files (and any illustration
                            images) into chapters/chNN_name/ folders
        │
        ▼
epub_builder/ module     → assembles chapters/ into a finished .epub
```

1. **[`ocr/`](ocr/README.md)** — batch-OCRs scanned pages into per-page
   `.txt` files. Several backends are available (a multimodal LLM via any
   OpenAI-compatible API is the recommended default; Google Cloud Vision
   and a fully offline `manga-ocr` pipeline are also included).
2. **Manual sorting** — split the resulting pages into chapter folders
   (`ch00_frontmatter/`, `ch01_chapter00/`, ...), optionally dropping in a
   `cover.jpg` and illustration images alongside the `.txt` files. This
   step is manual because automatically detecting chapter boundaries from
   OCR'd headers turned out to be unreliable — folder structure is simple
   and unambiguous instead.
3. **[`epub_builder/`](epub_builder/README.md)** — assembles the sorted
   `chapters/` folder into a valid `.epub`: furigana notation becomes
   `<ruby>` markup, images are placed inline, a font from `font/` gets
   embedded, and metadata comes from `config.json`.

See each module's own README for setup and usage details.

## Quick start

```bash
# 1. OCR
cd ocr
pip install openai pillow natsort tqdm
cp config.example.json config.json   # fill in api_key / base_url / model
python openrouter_ocr.py --input /path/to/scans --output ./out

# 2. Sort ./out/pages_txt/*.txt by hand into epub_builder/chapters/chNN_name/

# 3. Build the epub
cd ../epub_builder
pip install natsort
cp config.example.json config.json   # fill in title / author / ...
python build_epub.py
```

## ⚠️ A note on copyright

This repository contains only the **tooling**. It is not meant to, and
should not, be used to host or distribute:
- scanned page images,
- OCR'd text extracted from a copyrighted book,
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
