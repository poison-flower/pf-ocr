# EPUB builder module

Assembles a `chapters/` folder — laid out by hand into per-chapter
subfolders — into a finished, valid `.epub`, with furigana rendered as
proper `<ruby>` markup, an embedded font, a cover image, and metadata from
the `epub` section of the repo's root `config.json`.

This is step 2 of the pipeline; step 1 is the [`ocr`](../ocr/README.md)
module, which produces the per-page `.txt` files you'll sort into
`chapters/`.

## Layout

```
config.json                  <- shared with the ocr module; see below for the "epub" section
epub_builder/
  chapters/
    cover.jpg                <- cover image (optional, any "cover.*" directly in chapters/)
    ch00_frontmatter/        <- everything before the main text (title page, TOC, ...)
      001.txt
      002.jpg
    ch01_chapter00/
      003.txt
      004.txt
      005.jpg
    ch02_chapter01/
      006.txt
    ch08_backmatter/         <- everything after the main text (afterword, ...)
      ...
  font/
    YourFont.ttf              <- font to embed (optional)
  build_epub.py
  furigana.py
```

## Chapter folder naming: `chNN_name`

- `chNN` — sequence number, determines sort order (`ch00` → `ch01` → ... →
  `ch10`; use leading zeros if you have more than 9 chapters so plain
  alphabetical sort stays correct).
- `name` — default chapter title (override via `config.json`'s
  `epub.chapter_titles`).
- Folders with `frontmatter` or `backmatter` anywhere in their name are
  **always** ignored: no title, no TOC entry — regardless of any flag or
  config setting. Their content is still included in the book, just
  without a section label.

Files inside a folder are sorted strictly by the number in the filename
(extension doesn't matter — an image `004.jpg` will land between
`003.txt` and `005.txt`).

## Page layout

- The first line of each `.txt` page (after stripping a leading page
  number) is rendered as a larger, bold "page header" — typically an
  in-story date/time/chapter marker (relevant if you're OCR-ing a novel
  where that detail matters to the plot, e.g. one involving time travel).
- If nothing is left after stripping the page number, two blank lines are
  shown instead, so the page's vertical rhythm stays consistent.
- A blank line always follows the header, then the text.
- Each paragraph starts with a full-width space (`　`) — the standard
  indent in Japanese typography, embedded directly in the text rather than
  relying on CSS `text-indent` (which not every reader honors).

## config.json

This module reads the `epub` section of the shared `config.json` at the
repo root (copy `config.example.json` there if you haven't already):

```json
{
  "epub": {
    "title": "Your Book Title",
    "author": "Author Name",
    "language": "ja",
    "identifier": "",
    "publisher": "",
    "description": "",
    "date": "",
    "rights": "",
    "series": "",
    "series_index": "",
    "font_family": "",
    "vertical": false,
    "show_chapter_titles": false,
    "output": "book.epub",
    "chapter_titles": {
      "ch00_frontmatter": "",
      "ch01_chapter00": "Chapter 00",
      "ch08_backmatter": ""
    }
  }
}
```

- `title`, `author`, `language` — core metadata (`title` is required).
- `identifier`, `publisher`, `description`, `date`, `rights` — optional
  Dublin Core metadata, only added if non-empty. Leave `identifier` empty
  to get a random UUID generated on every build; set it once and keep it
  stable if you plan to rebuild the file repeatedly and want reading apps
  to recognize it as the same book.
- `series` / `series_index` — Calibre-style series metadata
  (`calibre:series` / `calibre:series_index`), recognized by Calibre,
  KOReader, and several other reading apps.
- `font_family` — explicit name for the embedded font family (see "Font"
  below). Optional — without it, the name is guessed automatically.
- `vertical` — vertical Japanese writing-mode (also toggleable via
  `--vertical`).
- `show_chapter_titles` — whether to print the chapter title (`<h1>`) at
  the start of each chapter. Defaults to `false` — titles still show up in
  the table of contents either way. Toggle via `--show-chapter-titles`.
- `output` — path for the resulting epub, used if `--output` isn't passed.
- `chapter_titles` — folder name -> display title mapping. An empty string
  `""` means "no heading, no TOC entry" (this is already the default
  behavior for `frontmatter`/`backmatter` folders, so listing them here is
  optional, just for clarity).

`frontmatter`/`backmatter` entries in `chapter_titles` are ignored no
matter what — they can't be "turned back on" via config.

## Font

Drop font file(s) (`.ttf`/`.otf`/`.woff`/`.woff2`) into `font/`. Name them
`Regular.ttf` / `Bold.ttf` / `Italic.ttf` / `BoldItalic.ttf` (case
doesn't matter, and the keyword can be part of a longer name, e.g.
`NotoSerifJP-Bold.ttf`) and they'll automatically be unified under one
font-family name with the correct `font-weight`/`font-style`. That means a
bold chapter heading (`<h1>`, and the page-header line) will automatically
pick up `Bold.ttf`, and italics (`<em>`), if any ever show up in the text,
will pick up `Italic.ttf`.

If the filenames don't match this pattern, the old behavior kicks in:
each file gets its own font-family name, and only the first one
(alphabetically) is wired up automatically in `body { font-family }`; the
rest are still embedded in the epub but need manual CSS edits to use.

The script tries to guess a shared family name from the common part of the
filenames (e.g. `NotoSerifJP-Regular.ttf` + `NotoSerifJP-Bold.ttf` ->
`NotoSerifJP`). To set it explicitly, use `config.json`'s `epub` section:

```json
{ "epub": { "font_family": "MyBookFont" } }
```

**Check the font's license** before embedding it in a file you intend to
share or publish — personal use on your own devices is usually fine, but
redistribution terms vary by font.

## Running

Simplest case — no arguments, if everything is where it's expected
(`config.json` one level up, at the repo root):

```bash
python build_epub.py
```

Or with explicit paths/overrides:

```bash
python build_epub.py \
  --pages-dir ./chapters \
  --font-dir ./font \
  --config ../config.json \
  --output ./my_book.epub \
  --show-chapter-titles \
  --vertical
```

`--split-pages` makes each file its own xhtml document inside the epub,
instead of merging every file in a folder into one.

## `furigana.py` as a standalone preview tool

Useful for quickly checking how a single OCR'd page will render before
running the full build:

```bash
python furigana.py --input page.txt --output preview.xhtml
# or a quick one-liner check:
echo "本文《ほんぶん》デザイン" | python furigana.py --stdin
```

## Requirements

```bash
pip install natsort
```
