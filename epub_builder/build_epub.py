#!/usr/bin/env python3
"""
Builds an EPUB from a chapters/ folder laid out by hand into per-chapter
subfolders, embedding a font from font/ and metadata from config.json.

Expected layout (next to this script):

    epub_builder/
      chapters/
        cover.jpg                <- cover image (optional, any "cover.*" directly in chapters/)
        ch00_frontmatter/        <- everything before the main text
          001.txt
          002.jpg
        ch01_chapter00/
          003.txt
          004.txt
          005.jpg
        ch02_chapter01/
          006.txt
        ch08_backmatter/
          ...
      font/
        YourFont.ttf              <- font to embed (optional)
      build_epub.py
      furigana.py

    config.json (at the repo root, shared with the ocr module — has an
    "epub" section with metadata and build settings)

Chapter folder naming: chNN_name
    - chNN determines sort order.
    - name is the default chapter title (can be overridden via
      "chapter_titles" in config.json).
    - Folders whose name (as a whole, or the part after chNN_) contains
      "frontmatter" or "backmatter" are ALWAYS ignored: no title, no TOC
      entry, no in-body heading — regardless of any flag or config
      setting. Their content is still included in the book, just without
      a section label.

Page layout:
    - The first line of each .txt page (after stripping a leading page
      number) is rendered as a larger, bold "page header" — usually an
      in-story date/time/chapter marker, not just page-numbering noise.
    - If nothing is left after stripping the page number, two blank lines
      are shown instead, so the page's vertical rhythm stays consistent.
    - A blank line always follows the header, then the text.
    - Each paragraph starts with a full-width space indent (standard in
      Japanese typography), not a tab character that HTML/EPUB would
      collapse anyway.

Chapter titles are shown in the table of contents by default, but NOT as
an in-body <h1> heading — pass --show-chapter-titles (or set
"show_chapter_titles": true in config.json) to also print them at the
start of each chapter.

Usage:
    python build_epub.py
    # (chapters/ and font/ are read next to this script; config.json is
    # read from the repo root, one level up)

    # or with explicit paths/overrides:
    python build_epub.py --pages-dir ./chapters --font-dir ./font \\
        --config ../config.json --output ./book.epub --show-chapter-titles

Requirements:
    pip install natsort
    (furigana.py must live in the same folder as this script)
"""

import argparse
import html
import json
import re
import sys
import uuid
import zipfile
from pathlib import Path

from natsort import natsorted

sys.path.insert(0, str(Path(__file__).resolve().parent))
from furigana import parse_page, page_to_paragraphs_html  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

TEXT_EXT = {".txt"}
IMAGE_EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
             ".webp": "image/webp", ".gif": "image/gif"}
FONT_EXT = {".ttf": "font/ttf", ".otf": "font/otf", ".woff": "font/woff", ".woff2": "font/woff2"}

CHAPTER_FOLDER_RE = re.compile(r"^ch(\d+)_(.+)$", re.IGNORECASE)

CSS_BASE = """
body {{ font-family: {font_stack}; line-height: 1.8; }}
body.vertical {{ writing-mode: vertical-rl; }}
p {{ margin: 0 0 1em 0; }}
p.page-header {{ font-size: 1.3em; font-weight: bold; margin: 1.5em 0 0 0; }}
p.spacer {{ margin: 0 0 1em 0; }}
rt {{ font-size: 0.5em; }}
h1 {{ text-align: center; margin: 2em 0; }}
div.illustration {{ text-align: center; margin: 1em 0; }}
div.illustration img {{ max-width: 100%; max-height: 100%; }}
div.cover {{ text-align: center; margin: 0; padding: 0; }}
div.cover img {{ max-width: 100%; height: auto; }}
{font_faces}
"""

FONT_FACE_TEMPLATE = """@font-face {{
  font-family: "{family}";
  src: url("../fonts/{filename}") format("{fmt}");
  font-weight: {weight};
  font-style: {style};
}}
"""

CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

CHAPTER_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head>
<meta charset="UTF-8"/>
<title>{title}</title>
<link rel="stylesheet" type="text/css" href="../css/style.css"/>
</head>
<body{body_class}>
{h1}
{content}
</body>
</html>
"""

COVER_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head>
<meta charset="UTF-8"/>
<title>Cover</title>
<link rel="stylesheet" type="text/css" href="../css/style.css"/>
</head>
<body>
<div class="cover"><img src="../{img_href}" alt="Cover"/></div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Page files: sorting, folder-name parsing
# ---------------------------------------------------------------------------

def numeric_sort_key(path: Path):
    """Sorts files strictly by the number found in the filename, ignoring the extension."""
    m = re.search(r"\d+", path.stem)
    if m:
        return (0, int(m.group()), path.name)
    return (1, 0, path.name)


def sort_files(files: list[Path]) -> list[Path]:
    return sorted(files, key=numeric_sort_key)


def parse_chapter_folder_name(folder_name: str) -> tuple[str | None, str]:
    """
    Parses a folder name of the form "chNN_name" -> (NN, name).
    Returns (None, folder_name) if it doesn't match the pattern.
    """
    m = CHAPTER_FOLDER_RE.match(folder_name)
    if m:
        return m.group(1), m.group(2)
    return None, folder_name


def is_ignored_section(folder_name: str) -> bool:
    """frontmatter/backmatter sections are always ignored, regardless of any flag."""
    normalized = folder_name.lower().replace("_", "").replace("-", "")
    return "frontmatter" in normalized or "backmatter" in normalized


def resolve_title(folder_name: str, chapter_titles_map: dict) -> str | None:
    """
    Returns the display title for a chapter, or None if the section should
    have no heading and no TOC entry (frontmatter/backmatter, or an
    explicit empty string in chapter_titles_map).
    """
    if is_ignored_section(folder_name):
        return None

    if folder_name in chapter_titles_map:
        value = chapter_titles_map[folder_name].strip()
        return value if value else None

    _, remainder = parse_chapter_folder_name(folder_name)
    default_title = remainder.replace("_", " ").strip()
    return default_title if default_title else folder_name


# ---------------------------------------------------------------------------
# Processing the files in one chapter folder (text + images)
# ---------------------------------------------------------------------------

def build_doc_from_folder(files: list[Path], doc_id: str):
    """Returns (content_html, [(arcname, filepath, media_type), ...])."""
    content_parts = []
    images = []

    for fpath in files:
        suffix = fpath.suffix.lower()
        if suffix in TEXT_EXT:
            text = fpath.read_text(encoding="utf-8")
            header, paragraphs = parse_page(text)
            if not header and not paragraphs:
                continue  # [NO_TEXT] / empty page
            content_parts.append(page_to_paragraphs_html(header, paragraphs))
        elif suffix in IMAGE_EXT:
            arcname = f"images/{doc_id}_{fpath.stem}{suffix}"
            content_parts.append(f'<div class="illustration"><img src="../{arcname}" alt=""/></div>')
            images.append((arcname, fpath, IMAGE_EXT[suffix]))
        else:
            print(f"Skipping unknown file type: {fpath}", file=sys.stderr)

    return "\n".join(content_parts), images


def find_cover(pages_dir: Path, explicit_cover: str | None) -> Path | None:
    if explicit_cover:
        p = Path(explicit_cover)
        return p if p.exists() else None
    for ext in IMAGE_EXT:
        candidates = list(pages_dir.glob(f"cover{ext}")) + list(pages_dir.glob(f"cover{ext.upper()}"))
        if candidates:
            return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Grouping pages into documents (chapters), with the split-pages option
# ---------------------------------------------------------------------------

def group_pages_into_docs(pages_dir: Path, cover_path: Path | None, chapter_titles_map: dict, split_pages: bool):
    subdirs = natsorted([d for d in pages_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    root_files = [
        p for p in pages_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (TEXT_EXT | set(IMAGE_EXT)) and p != cover_path
    ]

    if not subdirs and not root_files:
        print(f"No subfolders or page files found in {pages_dir}.", file=sys.stderr)
        sys.exit(1)

    docs = []
    all_images = []
    doc_counter = 0

    def add_doc_merged(folder_name: str, files: list[Path]):
        nonlocal doc_counter
        if not files:
            return
        doc_counter += 1
        doc_id = f"doc{doc_counter:03d}_{re.sub(r'[^a-zA-Z0-9]+', '', folder_name) or 'section'}"
        content, images = build_doc_from_folder(files, doc_id)
        if not content:
            doc_counter -= 1
            return
        title = resolve_title(folder_name, chapter_titles_map)
        docs.append({"id": doc_id, "title": title, "content": content})
        all_images.extend(images)

    def add_docs_split(folder_name: str, files: list[Path]):
        nonlocal doc_counter
        if not files:
            return
        folder_title = resolve_title(folder_name, chapter_titles_map)
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "", folder_name) or "section"
        first_page_in_folder = True
        for fpath in files:
            doc_counter += 1
            doc_id = f"doc{doc_counter:03d}_{safe_name}_{fpath.stem}"
            content, images = build_doc_from_folder([fpath], doc_id)
            if not content:
                doc_counter -= 1
                continue
            title = folder_title if first_page_in_folder else None
            first_page_in_folder = False
            docs.append({"id": doc_id, "title": title, "content": content})
            all_images.extend(images)

    add_doc = add_docs_split if split_pages else add_doc_merged

    if root_files:
        add_doc("root", sort_files(root_files))

    for subdir in subdirs:
        files = sort_files(
            [p for p in subdir.iterdir() if p.is_file() and p.suffix.lower() in (TEXT_EXT | set(IMAGE_EXT))]
        )
        add_doc(subdir.name, files)

    return docs, all_images


# ---------------------------------------------------------------------------
# Font handling
# ---------------------------------------------------------------------------

def find_fonts(font_dir: Path) -> list[Path]:
    if not font_dir.exists():
        return []
    fonts = [p for p in font_dir.iterdir() if p.is_file() and p.suffix.lower() in FONT_EXT]
    return natsorted(fonts, key=lambda p: p.name)


def sanitize_family_name(stem: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 _-]+", "", stem).strip() or "EmbeddedFont"


STYLE_KEYWORDS_RE = re.compile(r"(?i)bold[\s_-]*italic|italic[\s_-]*bold|bold|italic|oblique|regular")


def detect_font_role(stem: str) -> str | None:
    """
    Detects a font variant from its filename: Regular/Bold/Italic/BoldItalic
    (case-insensitive, anywhere in the name). Returns None if it can't be
    determined (non-standard filename).
    """
    lower = stem.lower()
    has_bold = "bold" in lower
    has_italic = "italic" in lower or "oblique" in lower
    if has_bold and has_italic:
        return "bolditalic"
    if has_bold:
        return "bold"
    if has_italic:
        return "italic"
    if "regular" in lower:
        return "regular"
    return None


ROLE_WEIGHT_STYLE = {
    "regular": ("normal", "normal"),
    "bold": ("bold", "normal"),
    "italic": ("normal", "italic"),
    "bolditalic": ("bold", "italic"),
}


def strip_style_keyword(stem: str) -> str:
    """Removes style keywords from a font filename, returning the remainder (may be empty)."""
    stripped = STYLE_KEYWORDS_RE.sub("", stem)
    stripped = re.sub(r"[\s_-]+", " ", stripped).strip(" -_")
    return stripped


def resolve_font_plan(fonts: list[Path], explicit_family: str | None):
    """
    Decides how to wire up the fonts found in font/:
    - If every file's variant is recognized (Regular/Bold/Italic/BoldItalic,
      with at least one Regular among them), all of them get ONE shared
      font-family name with the correct font-weight/font-style, so
      <b>/<strong>/<em>/bold headings automatically pick up the right file.
    - Otherwise (non-standard filenames): the old behavior — each file gets
      its own family name, and only the first one (alphabetically) gets
      wired up automatically in the CSS.

    Returns a list of tuples (font_path, family, weight, style, is_primary).
    """
    if not fonts:
        return []

    roles = {f: detect_font_role(f.stem) for f in fonts}
    all_recognized = all(r is not None for r in roles.values())
    has_regular = any(r == "regular" for r in roles.values())

    if all_recognized and has_regular:
        if explicit_family:
            family = explicit_family
        else:
            stripped_names = [strip_style_keyword(f.stem) for f in fonts]
            non_empty = [n for n in stripped_names if n]
            family = max(set(non_empty), key=non_empty.count) if non_empty else "MainFont"
            family = sanitize_family_name(family) or "MainFont"

        plan = []
        for f in fonts:
            weight, style = ROLE_WEIGHT_STYLE[roles[f]]
            is_primary = roles[f] == "regular"
            plan.append((f, family, weight, style, is_primary))
        return plan

    # Fallback: non-standard filenames — separate families
    plan = []
    for i, f in enumerate(fonts):
        family = explicit_family if (explicit_family and i == 0) else sanitize_family_name(f.stem)
        plan.append((f, family, "normal", "normal", i == 0))
    return plan


# ---------------------------------------------------------------------------
# Assembling the final epub
# ---------------------------------------------------------------------------

def build_epub(docs, all_images, cover_path, fonts: list[Path], output_path: Path,
               meta: dict, vertical: bool, show_chapter_titles: bool):
    book_id = meta.get("identifier") or f"urn:uuid:{uuid.uuid4()}"
    body_class = ' class="vertical"' if vertical else ""

    manifest_items = []
    spine_items = []
    nav_items = []
    package_files = {}

    # --- Font ---
    font_faces_css = ""
    font_family_stack = "serif"
    if fonts:
        font_plan = resolve_font_plan(fonts, meta.get("font_family"))
        font_face_rules = []
        primary_family = None
        variant_labels = []

        for font_path, family, weight, style, is_primary in font_plan:
            fmt = FONT_EXT[font_path.suffix.lower()].split("/")[-1]
            font_face_rules.append(
                FONT_FACE_TEMPLATE.format(family=family, filename=font_path.name, fmt=fmt, weight=weight, style=style)
            )
            arcname = f"fonts/{font_path.name}"
            package_files[f"OEBPS/{arcname}"] = font_path.read_bytes()
            font_id = "font_" + re.sub(r"[^a-zA-Z0-9]+", "_", font_path.name)
            manifest_items.append(f'<item id="{font_id}" href="{arcname}" media-type="{FONT_EXT[font_path.suffix.lower()]}"/>')
            variant_labels.append(f"{font_path.name} ({weight}/{style})")
            if is_primary and primary_family is None:
                primary_family = family

        font_faces_css = "\n".join(font_face_rules)
        font_family_stack = f'"{primary_family}", serif'

        unified = len(set(family for _, family, *_ in font_plan)) == 1 and len(font_plan) > 1
        if unified:
            print(f'Font: {len(fonts)} variant(s) unified under the name "{primary_family}": '
                  + ", ".join(variant_labels))
        elif len(fonts) > 1:
            print(f"Found {len(fonts)} font file(s). Filenames don't look like Regular/Bold/Italic — "
                  f"using {fonts[0].name} as the primary one. The rest are embedded in the epub but "
                  f"not wired up in the CSS automatically — edit style.css by hand if needed.")

    css_content = CSS_BASE.format(font_stack=font_family_stack, font_faces=font_faces_css)

    # --- Cover ---
    cover_meta = ""
    guide_xml = ""
    if cover_path:
        cover_ext = cover_path.suffix.lower()
        cover_media = IMAGE_EXT.get(cover_ext, "image/jpeg")
        cover_img_arcname = f"images/cover{cover_ext}"
        package_files[f"OEBPS/{cover_img_arcname}"] = cover_path.read_bytes()
        manifest_items.append(
            f'<item id="cover-image" href="{cover_img_arcname}" media-type="{cover_media}" properties="cover-image"/>'
        )
        package_files["OEBPS/text/cover.xhtml"] = COVER_TEMPLATE.format(img_href=cover_img_arcname)
        manifest_items.append('<item id="cover-page" href="text/cover.xhtml" media-type="application/xhtml+xml"/>')
        spine_items.append('<itemref idref="cover-page" linear="yes"/>')
        cover_meta = '<meta name="cover" content="cover-image"/>'
        guide_xml = '<guide><reference type="cover" title="Cover" href="text/cover.xhtml"/></guide>'

    # --- Images from chapters ---
    for arcname, filepath, media_type in all_images:
        package_files[f"OEBPS/{arcname}"] = filepath.read_bytes()
        img_id = "img_" + re.sub(r"[^a-zA-Z0-9]+", "_", arcname)
        manifest_items.append(f'<item id="{img_id}" href="{arcname}" media-type="{media_type}"/>')

    # --- Chapters ---
    for doc in docs:
        fname = f"text/{doc['id']}.xhtml"
        display_title = doc["title"]
        xhtml_title = display_title if display_title else " "
        h1 = f"<h1>{html.escape(display_title)}</h1>" if (display_title and show_chapter_titles) else ""
        package_files[f"OEBPS/{fname}"] = CHAPTER_TEMPLATE.format(
            title=html.escape(xhtml_title), body_class=body_class, h1=h1, content=doc["content"],
        )
        manifest_items.append(f'<item id="{doc["id"]}" href="{fname}" media-type="application/xhtml+xml"/>')
        spine_items.append(f'<itemref idref="{doc["id"]}"/>')
        if display_title:
            nav_items.append(f'<li><a href="{fname}">{html.escape(display_title)}</a></li>')

    manifest_extra = (
        '<item id="css" href="css/style.css" media-type="text/css"/>\n'
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>\n'
        '    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
    )

    # --- Additional dc: metadata ---
    dc_extra = []
    if meta.get("publisher"):
        dc_extra.append(f"<dc:publisher>{html.escape(meta['publisher'])}</dc:publisher>")
    if meta.get("description"):
        dc_extra.append(f"<dc:description>{html.escape(meta['description'])}</dc:description>")
    if meta.get("date"):
        dc_extra.append(f"<dc:date>{html.escape(meta['date'])}</dc:date>")
    if meta.get("rights"):
        dc_extra.append(f"<dc:rights>{html.escape(meta['rights'])}</dc:rights>")
    if meta.get("series"):
        dc_extra.append(f'<meta name="calibre:series" content="{html.escape(meta["series"])}"/>')
        if meta.get("series_index"):
            dc_extra.append(f'<meta name="calibre:series_index" content="{html.escape(str(meta["series_index"]))}"/>')

    content_opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="BookId" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="BookId">{book_id}</dc:identifier>
    <dc:title>{html.escape(meta['title'])}</dc:title>
    <dc:creator>{html.escape(meta['author'])}</dc:creator>
    <dc:language>{meta['language']}</dc:language>
    {cover_meta}
    {chr(10).join('    ' + line for line in dc_extra)}
  </metadata>
  <manifest>
    {manifest_extra}
    {chr(10).join('    ' + item for item in manifest_items)}
  </manifest>
  <spine>
    {chr(10).join('    ' + item for item in spine_items)}
  </spine>
  {guide_xml}
</package>
"""

    nav_xhtml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="{meta['language']}">
<head><meta charset="UTF-8"/><title>Table of Contents</title></head>
<body>
<nav epub:type="toc" id="toc">
<h1>Table of Contents</h1>
<ol>
{chr(10).join(nav_items)}
</ol>
</nav>
</body>
</html>
"""

    nav_points = []
    playorder = 0
    for doc in docs:
        if not doc["title"]:
            continue
        playorder += 1
        nav_points.append(f"""    <navPoint id="navpoint-{playorder}" playOrder="{playorder}">
      <navLabel><text>{html.escape(doc['title'])}</text></navLabel>
      <content src="text/{doc['id']}.xhtml"/>
    </navPoint>""")

    toc_ncx = f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="{book_id}"/></head>
  <docTitle><text>{html.escape(meta['title'])}</text></docTitle>
  <navMap>
{chr(10).join(nav_points)}
  </navMap>
</ncx>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", content_opf)
        zf.writestr("OEBPS/nav.xhtml", nav_xhtml)
        zf.writestr("OEBPS/toc.ncx", toc_ncx)
        zf.writestr("OEBPS/css/style.css", css_content)
        for arcname, content in package_files.items():
            zf.writestr(arcname, content)

    print(f"EPUB built: {output_path}")
    print(f"Total sections: {len(docs)}")
    for doc in docs:
        label = doc["title"] if doc["title"] else "(untitled, not in TOC)"
        print(f"  - {doc['id']}: {label}")
    if cover_path:
        print(f"Cover: {cover_path}")
    if fonts:
        print(f"Fonts embedded: {len(fonts)} (primary: {fonts[0].name})")


# ---------------------------------------------------------------------------
# Config and entry point
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Loads the shared config.json and returns its "epub" section.

    Falls back to treating the whole file as the epub config if there's no
    "epub" key, so a bare {"title": ...} style file still works.
    """
    if not config_path.exists():
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return data.get("epub", data) if isinstance(data, dict) else {}


def main():
    parser = argparse.ArgumentParser(description="Build an EPUB from a chapters/ folder, with metadata from config.json")
    parser.add_argument("--pages-dir", default=str(SCRIPT_DIR / "chapters"), help="Folder with chapter subfolders")
    parser.add_argument("--font-dir", default=str(SCRIPT_DIR / "font"), help="Folder with the font to embed")
    parser.add_argument(
        "--config", default=str(ROOT_DIR / "config.json"),
        help="Path to config.json with an \"epub\" section. Defaults to config.json at the repo root."
    )
    parser.add_argument("--output", default=None, help="Path to the resulting .epub (defaults to config.json's value)")
    parser.add_argument("--cover", default=None, help="Explicit cover image path (overrides auto-detection and config.json)")
    parser.add_argument("--title", default=None, help="Override title from config.json")
    parser.add_argument("--author", default=None, help="Override author from config.json")
    parser.add_argument("--language", default=None, help="Override language from config.json")
    parser.add_argument("--vertical", action="store_true", default=None, help="Vertical Japanese writing-mode")
    parser.add_argument(
        "--show-chapter-titles", action="store_true", default=None,
        help="Print the chapter title (<h1>) at the start of each chapter. By default it only appears in the TOC."
    )
    parser.add_argument(
        "--split-pages", action="store_true",
        help="Make each file its own xhtml document, instead of merging all files in a folder into one."
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    meta = {
        "title": args.title or config.get("title"),
        "author": args.author or config.get("author", "Unknown"),
        "language": args.language or config.get("language", "ja"),
        "identifier": config.get("identifier") or None,
        "font_family": config.get("font_family") or None,
        "publisher": config.get("publisher"),
        "description": config.get("description"),
        "date": config.get("date"),
        "rights": config.get("rights"),
        "series": config.get("series"),
        "series_index": config.get("series_index"),
    }
    if not meta["title"]:
        print("No title set — put one in config.json or pass --title.", file=sys.stderr)
        sys.exit(1)

    vertical = args.vertical if args.vertical is not None else bool(config.get("vertical", False))
    show_chapter_titles = (
        args.show_chapter_titles if args.show_chapter_titles is not None
        else bool(config.get("show_chapter_titles", False))
    )
    chapter_titles_map = config.get("chapter_titles", {})

    output_path = Path(args.output) if args.output else Path(config.get("output", str(SCRIPT_DIR / "book.epub")))

    pages_dir = Path(args.pages_dir)
    font_dir = Path(args.font_dir)

    explicit_cover = args.cover or config.get("cover")
    cover_path = find_cover(pages_dir, explicit_cover)
    fonts = find_fonts(font_dir)

    docs, all_images = group_pages_into_docs(pages_dir, cover_path, chapter_titles_map, args.split_pages)

    build_epub(
        docs, all_images, cover_path, fonts, output_path,
        meta=meta, vertical=vertical, show_chapter_titles=show_chapter_titles,
    )


if __name__ == "__main__":
    main()
