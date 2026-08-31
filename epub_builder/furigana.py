#!/usr/bin/env python3
"""
Shared logic for turning OCR page text (see ocr/prompt.txt for the expected
format) into HTML paragraphs, including furigana rendered as proper <ruby>
markup. Used by build_epub.py, and also runnable standalone as a quick
preview/debugging tool for a single page.

Input text format (as produced by the OCR module):
    - First line: page header/date/chapter-marker metadata.
    - Every following line: one paragraph (no blank-line separators).

Furigana notation on input (Aozora Bunko style):
    - base《reading》   Furigana over `base`. `base` is the nearest
                        contiguous run of kanji immediately before《.
    - ｜base《reading》 Explicit start of `base`, needed when it doesn't
                        match a plain "kanji run" (e.g. it's shorter, or
                        contains non-kanji characters, or is glued to the
                        previous word without a natural boundary).
    - Several ｜base《reading》 in a row are kept as separate <ruby> groups,
      in reading order (e.g. a compound word whose furigana was printed
      split across its parts).

Standalone usage:
    python furigana.py --input page.txt --output page.xhtml
    # or a quick one-off check:
    echo "本文《ほんぶん》デザイン" | python furigana.py --stdin
"""

import argparse
import html
import re
import sys
from pathlib import Path

# Kanji range + the iteration mark 々 + a couple of CJK compatibility ranges
KANJI_RUN = r"[\u4e00-\u9fff\u3005\u3007\uf900-\ufaff]+"

# 1) Explicit base boundary via ｜: ｜<anything but ｜《>《reading》
RE_MARKED = re.compile(r"｜([^｜《]+?)《([^》]+)》")
# 2) No ｜: base is the nearest kanji run right before《
RE_AUTO = re.compile(r"(" + KANJI_RUN + r")《([^》]+)》")

# Strips a leading page number like "5 " or "23 " from the page header
LEADING_PAGE_NUMBER_RE = re.compile(r"^\d+\s*")

# Full-width space — the standard paragraph indent in Japanese typography
PARAGRAPH_INDENT = "\u3000"

XHTML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head>
<meta charset="UTF-8"/>
<title>{title}</title>
<style type="text/css">
{extra_style}  p {{ margin: 0 0 1em 0; }}
  p.page-header {{ font-size: 1.3em; font-weight: bold; margin: 1.5em 0 0 0; }}
  p.spacer {{ margin: 0 0 1em 0; }}
  rt {{ font-size: 0.5em; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""

VERTICAL_STYLE = "  body { writing-mode: vertical-rl; font-family: serif; }\n"
HORIZONTAL_STYLE = "  body { font-family: serif; }\n"


def furigana_to_ruby(text: str) -> str:
    """Replaces Aozora Bunko furigana notation with <ruby><rb>...</rb><rt>...</rt></ruby>."""
    def repl(m: re.Match) -> str:
        base, reading = m.group(1), m.group(2)
        return f"<ruby><rb>{base}</rb><rt>{reading}</rt></ruby>"

    text = RE_MARKED.sub(repl, text)
    text = RE_AUTO.sub(repl, text)
    return text


def parse_page(text: str) -> tuple[str, list[str]]:
    """Returns (header, [paragraphs]) from a page's raw OCR text.

    The literal marker "[NO_TEXT]" (written by the OCR module for pages
    with no text at all, e.g. illustrations) parses as an empty page.
    """
    if text.strip() == "[NO_TEXT]":
        return "", []
    lines = [ln.strip() for ln in text.strip("\n").split("\n") if ln.strip() != ""]
    if not lines:
        return "", []
    return lines[0], lines[1:]


def strip_leading_page_number(header: str) -> str:
    return LEADING_PAGE_NUMBER_RE.sub("", header, count=1)


def page_to_paragraphs_html(header: str, paragraphs: list[str]) -> str:
    """
    Renders one page's (header, paragraphs) as HTML <p> elements:
      - The header (with its leading page number stripped) is shown in a
        larger, bold "page-header" paragraph — in the source novels this
        line is usually an in-story date/time/chapter marker, not just
        page-numbering noise.
      - If the header is empty after stripping the page number, an empty
        placeholder line is rendered instead, so the vertical rhythm of the
        page stays consistent whether or not there was a header.
      - A blank spacer line always follows the header.
      - Each paragraph is prefixed with a full-width space (the standard
        Japanese paragraph indent) rather than relying on CSS text-indent,
        so the indent survives even in readers that ignore that CSS rule.
    """
    header = strip_leading_page_number(header).strip()

    header_html = furigana_to_ruby(html.escape(header, quote=False)) if header else "&#160;"

    out = [
        f'<p class="page-header">{header_html}</p>',
        '<p class="spacer">&#160;</p>',
    ]

    for para in paragraphs:
        indented = PARAGRAPH_INDENT + para
        escaped = html.escape(indented, quote=False)
        out.append(f"<p>{furigana_to_ruby(escaped)}</p>")

    return "\n".join(out)


def convert_page(raw_text: str) -> str:
    """Convenience wrapper: raw OCR page text -> HTML <p> fragment (no <html>/<body> wrapper)."""
    header, paragraphs = parse_page(raw_text)
    if not header and not paragraphs:
        return ""
    return page_to_paragraphs_html(header, paragraphs)


def main():
    parser = argparse.ArgumentParser(description="Convert Aozora-style furigana notation to XHTML <ruby>")
    parser.add_argument("--input", help="Path to a text file (one page's OCR output)")
    parser.add_argument("--output", help="Where to write the XHTML (defaults to stdout)")
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin (for quick checks)")
    parser.add_argument(
        "--fragment", action="store_true",
        help="Output only the <p> fragment, without wrapping it in a full XHTML document. "
             "Useful when the fragment will be inserted into an assembled chapter file elsewhere."
    )
    parser.add_argument("--title", default="page", help="Document title (for --output, not --fragment)")
    parser.add_argument(
        "--vertical", action="store_true",
        help="Vertical Japanese writing-mode (as in the original scan). "
             "Defaults to horizontal (left-to-right) text, which renders more reliably "
             "across readers and on mobile."
    )
    args = parser.parse_args()

    if args.stdin:
        raw_text = sys.stdin.read()
    elif args.input:
        raw_text = Path(args.input).read_text(encoding="utf-8")
    else:
        print("Provide --input <file> or --stdin", file=sys.stderr)
        sys.exit(1)

    body = convert_page(raw_text)

    if args.fragment:
        result = body
    else:
        extra_style = VERTICAL_STYLE if args.vertical else HORIZONTAL_STYLE
        result = XHTML_TEMPLATE.format(title=html.escape(args.title), body=body, extra_style=extra_style)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Saved: {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
