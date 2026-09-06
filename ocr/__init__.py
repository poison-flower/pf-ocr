"""
OCR module: batch-transcribes (or translates) scanned pages into per-page
.txt files. Light novel and manga pages are handled by separate scripts,
since their layouts need genuinely different logic (dense running prose
vs. scattered, typed speech bubbles that benefit from translation +
glossary + cross-page continuity context).

Novel (light novel, dense running prose):
- novel_ocr.py          Any OpenAI-compatible API (OpenRouter, a direct
                          provider endpoint, a self-hosted proxy, etc.)
                          with a multimodal model. Recommended.
- google_vision_ocr.py  Classic OCR via Google Cloud Vision (no LLM,
                          --mode novel).

Manga (speech bubbles / narration boxes / SFX):
- manga_ocr_llm.py      Any OpenAI-compatible API. Transcribe or
                          --translate, with --glossary and --context-pages
                          for cross-page continuity. Recommended.
- google_vision_ocr.py  Classic OCR via Google Cloud Vision (no LLM,
                          --mode manga; transcription only, no translation).
- local_mangaocr_ocr.py Fully offline, no cloud account, via manga-ocr
                          (also supports --mode novel via column
                          segmentation, transcription only).

Each script is self-contained and runnable directly, e.g.:
    python -m ocr.novel_ocr --input ./pages --output ./out
    python -m ocr.manga_ocr_llm --input ./pages --output ./out --translate
"""
