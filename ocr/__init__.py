"""
OCR module: batch-transcribes scanned light novel pages (vertical Japanese
text) into per-page .txt files, ready for the epub_builder module.

Several interchangeable backends are provided as standalone scripts:

- openrouter_ocr.py    Recommended. Any OpenAI-compatible API (OpenRouter,
                        a self-hosted proxy, etc.) with a multimodal model.
- gemini_direct_ocr.py Same idea, but calling the Gemini API directly.
- google_vision_ocr.py Classic OCR via Google Cloud Vision (no LLM).
- local_mangaocr_ocr.py Fully offline, no cloud account, via manga-ocr.

Each script is self-contained and runnable directly, e.g.:
    python -m ocr.openrouter_ocr --input ./pages --output ./out
"""
