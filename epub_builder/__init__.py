"""
EPUB builder module: assembles a chapters/ folder (per-page .txt files from
the ocr module, plus optional illustrations) into a finished, valid .epub —
with furigana rendered as proper <ruby> markup, an embedded font, a cover
image, and metadata pulled from config.json.

Modules:
    furigana.py    Aozora Bunko furigana notation -> HTML <ruby> markup,
                   plus the shared page-parsing/rendering logic. Also
                   runnable standalone as a single-page preview tool.
    build_epub.py  Assembles the whole book. Runnable directly:
                       python -m epub_builder.build_epub
"""
