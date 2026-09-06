# OCR module

Batch-transcribes (or translates) scanned pages into per-page `.txt`
files. Light novel and manga pages are handled by **separate scripts**,
since their layouts need genuinely different logic — dense running prose
vs. scattered speech bubbles that benefit from translation, a glossary,
and cross-page continuity context:

| | Light novel | Manga |
|---|---|---|
| Recommended script | [`novel_ocr.py`](#novel_ocrpy-recommended-for-light-novels) | [`manga_ocr_llm.py`](#manga_ocr_llmpy-recommended-for-manga) |
| Pure-OCR alternative | `google_vision_ocr.py --mode novel` | `google_vision_ocr.py --mode manga`, `local_mangaocr_ocr.py` |
| Layout assumed | Dense running prose, read top-to-bottom then right-to-left | Speech bubbles / narration boxes / SFX scattered across panels |
| Output | Continuous transcribed text per page | Numbered list, one entry per bubble, in manga reading order |
| Translation | — | `--translate`, with glossary + cross-page context |

For a light novel, this is step 1 of the pipeline — step 2 is
[`epub_builder`](../epub_builder/README.md), which turns the resulting
`.txt` files into a finished `.epub`. For manga, these `.txt` files are
meant as reference for your own typesetting workflow; `epub_builder`
targets prose light novels and doesn't lay out manga pages.

## `novel_ocr.py` (recommended for light novels)

Reads each page with a multimodal LLM via any OpenAI-compatible API
(OpenRouter, a direct provider endpoint, a self-hosted proxy, etc. —
including Gemini, GPT-4V-class models, or anything else exposed through
such an endpoint) and transcribes the vertical Japanese prose. Kept
deliberately simple — transcription only, no translate/glossary/context —
since that's genuinely all a novel page needs.

```bash
pip install openai pillow natsort tqdm
```

1. From the repo root, copy `config.example.json` to `config.json` and
   fill in the `ocr` section's `api_key`, `base_url` (your provider's
   OpenAI-compatible endpoint, usually ending in `/v1`), and `model`
   identifier. This same file is shared with `manga_ocr_llm.py` and the
   `epub_builder` module. `temperature`/`max_tokens`/`top_p`/
   `reasoning_effort` are optional — see [API parameters](#api-parameters).
2. Optionally edit `prompt_novel.txt` — plain English text, no need to
   touch any code to tweak the instructions given to the model.

```bash
python novel_ocr.py --input ./pages --output ./out
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
- `--config` overrides the config.json path (defaults to the repo root).
- `--prompt-file` overrides the prompt file (defaults to `prompt_novel.txt`).

## `manga_ocr_llm.py` (recommended for manga)

Same idea as `novel_ocr.py` — any OpenAI-compatible API, whole page in one
request — but built around what a manga page actually needs: bubbles
transcribed as a numbered, typed, reading-order list, optional direct
translation, a glossary for consistent names/terms, and continuity context
from previous pages.

```bash
pip install openai pillow natsort tqdm
```

Setup is identical to `novel_ocr.py` (same `config.json`, same `ocr`
section) — just point at `prompt_manga.txt` / `prompt_manga_translate.txt`
if you want to edit the instructions.

```bash
# transcribe
python manga_ocr_llm.py --input ./pages --output ./out

# translate directly instead
python manga_ocr_llm.py --input ./pages --output ./out --translate --target-lang Russian
```

Output format is a numbered list per page:

```
1. [DIALOGUE] ...
2. [SFX] ...
3. [NARRATION] ...
```

where entries are ordered manga-style (panels right-to-left top-to-bottom;
bubbles within a panel the same way), and TYPE is one of DIALOGUE,
THOUGHT, NARRATION, or SFX. Same `--input`/`--output`/`--sleep`/`--config`/
`--prompt-file` conventions as `novel_ocr.py`.

### Translation (`--translate`)

- `--target-lang`: written out as a plain language name (`Russian`,
  `English`, `Spanish`, ...) — it's dropped directly into the prompt as
  "translate all text on the page into {target-lang}". Defaults to
  `English`.
- This is a plain LLM translation, not a professional-quality localization
  pass — treat it as a strong first draft to edit, not a final one.
- Use a different `--output` folder than your transcription run — both
  write to `pages_txt/<name>.txt`, and the script skips a page whose
  output file already exists, so reusing the same folder would just
  return the old transcription instead of translating.

### Glossary

For a consistent translation of recurring names and terms, copy
[`glossary.example.md`](glossary.example.md) to `glossary.md` right here in
the `ocr/` folder (next to the scripts, not with your page images — a
single stable file you keep building up across books/chapters, which also
makes it a natural fit for an editing UI later). It's auto-detected and
sent along with every `--translate` request, e.g.:

```markdown
## Names
- 澤村・スペンサー・英梨々: Савамура Спенсер Эрири

## Terms
- ビジュアルノベル: Визуальная новелла

## Notes
- Keep Japanese suffixes (e.g. -chan, -san).
```

The file isn't parsed — it's appended to the prompt more or less as-is —
so feel free to add/reorder/rename sections, drop entries, or write extra
freeform notes for the model. Override the auto-detected path with
`--glossary /path/to/glossary.md` if you keep it somewhere else.

### Continuity context from previous pages

`--context-pages N` (default `2`) also sends along the translations of the
last N pages, so the model can keep character voice consistent and resolve
things that only make sense given what was just said — an ongoing
exchange, a pronoun referring back to something on the previous page, a
punchline that depends on the setup a page earlier. Manga bubbles are
terse and full of exactly this kind of dependency, which is why this
exists here and not in `novel_ocr.py`.

```bash
python manga_ocr_llm.py --input ./pages --output ./out --translate --context-pages 4
```

- Only kicks in with `--translate` (there's nothing to carry forward when
  just transcribing).
- Pulls from `pages_txt/` in your `--output` folder — the translations
  already produced earlier in this same run (or a previous run you're
  resuming). Blank/`[NO_TEXT]` pages are skipped when building context,
  since they add nothing.
- `--context-pages 0` disables it.
- Token cost is minimal — manga dialogue is short — but the log for each
  page (see below) records exactly which previous pages were included, if
  you want to check.

### Lookahead context from upcoming pages

`--context-pages-ahead N` (default `0`) goes the other direction: it sends
along the **original Japanese** (never a translation) of the next N pages,
for things that only make sense once you know what happens next — a
pronoun whose gender only becomes clear a page later, a line whose real
addressee is revealed afterward, a joke whose setup pays off on the
following page. Professional manga translators read a whole chapter before
translating any of it for exactly this reason.

```bash
python manga_ocr_llm.py --input ./pages --output ./out --translate --context-pages-ahead 1
```

**Deliberately the original Japanese, never a draft translation.** An
earlier version of this feature was going to run a cheap first-pass
*translation* of upcoming pages and feed that forward as context. Don't do
that — a translation is someone's (or something's) interpretation, not raw
fact, and even with a "treat this as an unreliable draft" instruction, a
model given a wrong reading in the context tends to partially inherit it.
A rushed, context-blind draft translation of the *next* page is exactly as
likely to be wrong as a rushed translation of the current one — so this
would be laundering a coin flip's worth of noise into looking like ground
truth. Raw Japanese has no such failure mode: it's just data, correctly
transcribed once and reused, and the model draws its own conclusions from
it exactly the same way it would from the current page.

- Only kicks in with `--translate`.
- Before translating a page, this makes sure the next N pages' Japanese
  text is available:
  1. **Hand-prepared or previously-cached `.txt`** — if `--context-src-dir`
     (default `<output>/context_src/`) already has a file named exactly
     like that page's image stem (e.g. `page005.txt` for `page005.jpg`),
     it's used as-is, no OCR call made. You can drop hand-corrected
     transcriptions in here yourself ahead of time if you want full
     control — nothing requires them to come from a script.
  2. **Otherwise, OCR'd on the fly** via `--ahead-ocr-backend` (`llm`
     default — the same OpenAI-compatible API, transcription prompt, no
     translation; or `local` — offline `manga-ocr`, needs the `manga-ocr`
     and `opencv-python` packages) and cached into `context_src/` for
     reuse — a page used as lookahead context is only ever OCR'd once,
     even though it'll come up again as the "current" page (or as another
     page's lookahead) later in the run.
- `--context-pages-ahead 0` (default) disables it entirely — no extra OCR
  calls, no `context_src/` folder created.
- Extra OCR calls mean extra latency/cost on pages that need lookahead —
  small in absolute terms (manga bubbles are short), but worth knowing
  it's there. The log for each page records exactly which upcoming pages'
  Japanese was included (`context_pages_ahead_used`).
- Combine freely with `--context-pages` — a page's prompt can include both
  past *translations* and future *Japanese* at once, kept in clearly
  separate, clearly labeled sections so the model doesn't confuse the two.

## API parameters

`novel_ocr.py` and `manga_ocr_llm.py` (the two OpenAI-compatible-API
scripts) read request parameters from the `ocr` section of `config.json`,
each overridable with a matching CLI flag:

| config.json field | CLI flag | Notes |
|---|---|---|
| `temperature` | `--temperature` | Default `0` (deterministic — you want the same page OCR'd the same way every time). |
| `max_tokens` | `--max-tokens` | Response length cap. Omitted from the request unless set. |
| `top_p` | `--top-p` | Omitted from the request unless set. |
| `reasoning_effort` | `--reasoning-effort` | e.g. `low`/`medium`/`high`. Passed through as `extra_body`, since support varies by model/provider — if your model/provider ignores it, it's simply a no-op rather than an error. |

`google_vision_ocr.py` doesn't take any of these (classic OCR, no model
parameters); `local_mangaocr_ocr.py` doesn't either (a fixed local model).

## Pure-OCR alternatives (no translation)

| Script | Backend | Setup needed | Notes |
|---|---|---|---|
| `google_vision_ocr.py` | Google Cloud Vision (classic OCR) | Google Cloud project + billing enabled | No LLM context understanding, but solid on clean scans. `--mode novel`/`--mode manga` (see below). Free tier covers a typical volume, but Google still requires a billing account to be linked. |
| `local_mangaocr_ocr.py` | [manga-ocr](https://github.com/kha-white/manga-ocr), fully offline | None — no account, no API key | `--mode novel` (column segmentation) or `--mode manga` (speech-bubble detection, closer to what manga-ocr was actually trained on). |

Both are transcription-only (no `--translate`) and follow the same
`--input`/`--output`/`--mode` convention as the LLM scripts, writing to
the same `pages_txt/*.txt` + `combined.md` layout. See each script's own
docstring for setup and manga-mode caveats (`local_mangaocr_ocr.py` in
particular — bubble detection is a geometric heuristic, so pass `--debug`
to sanity-check the detected reading order).

`google_vision_ocr.py` reads its `credentials` path from config.json's
**`vision`** section (not `ocr` — Google Cloud Vision has nothing to do
with the OpenAI-compatible API the other scripts use):

```json
{ "vision": { "credentials": "/path/to/service-account-key.json" } }
```

## Request/response logs

`novel_ocr.py`, `manga_ocr_llm.py`, and `google_vision_ocr.py` each write
one JSON log file per page to `logs/` at the repo root (created
automatically) — the full prompt/request sent, every retry attempt, and
the full raw API response (or the error, if it failed). Handy for
debugging a bad transcription/translation, checking token usage, or just
seeing exactly what was sent.

```
logs/
  20260901_153012_042311_page003.json
  20260901_153034_198822_page004.json
  ...
```

- The image itself is never embedded in the log (only its filename/size) —
  everything else about the request is recorded as-is.
- `--log-dir` points logging at a different folder; `--no-log` disables it.
- Logs accumulate across every run (nothing is deleted automatically) —
  clear out `logs/` periodically if it grows large.
- `local_mangaocr_ocr.py` makes no network requests (fully offline), so
  there's no "response" to log for it.

## Next step

Once you have a `pages_txt/` folder full of `.txt` files:
- **Light novel**: manually sort the pages into the folder structure
  `epub_builder` expects (see
  [`epub_builder/README.md`](../epub_builder/README.md)), then run the epub
  builder.
- **Manga**: these `.txt` files (one numbered bubble list per page) are
  meant as transcription/translation reference for your own typesetting
  workflow — `epub_builder` targets prose light novels and doesn't lay out
  manga pages.
