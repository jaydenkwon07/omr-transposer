# 0001 — Renderer: Verovio + rsvg-convert, behind an interface

Status: accepted (Project 0, 2026-07-09)

## Context

Project 0 must render transposed MusicXML to a PDF. CLAUDE.md fixes the shape: keep the
renderer behind a small interface so LilyPond can be swapped in later. The interface is:

```python
class Renderer(Protocol):
    def to_pdf(self, musicxml: str) -> bytes: ...
```

Verovio is the obvious engraver — it is fast, pip-installable, and reads MusicXML — but
it only emits **SVG** (and MIDI/MEI). Something has to turn SVG into PDF and stitch
multiple pages. The initial plan named `cairosvg` for that step because it is a
pure-Python API. Verovio draws SMuFL music glyphs with `<use>` references into a
`<defs>` block, which is exactly the SVG feature a converter is most likely to get wrong,
so we spiked it before committing.

## What the spike found

On a Bach chorale (`bach/bwv66.6`), Verovio produced a clean 157 KB SVG using `<use>`
glyph references. Then:

- **cairosvg** does not import at all on this machine: `cairocffi` cannot locate
  Homebrew's `libcairo` by leaf name because `/opt/homebrew/lib` is not on the macOS
  loader path. It honors `DYLD_LIBRARY_PATH` / `DYLD_FALLBACK_LIBRARY_PATH`, but those
  must be set **before** the process starts — dyld caches them at launch. A `ctypes`
  preload of the full dylib path did not help; `cairocffi` re-resolves by leaf name. A
  console-script entry point (`omrt = ...:main`) cannot reliably set this, so cairosvg
  would make the CLI fragile-by-default and hard to package.
- **rsvg-convert** (from `librsvg`) is a standalone binary that resolves its own
  libraries via rpath, so it has none of the loader-path problem. It rendered the SVG to
  both PDF and PNG; the PNG showed correct treble/bass clefs, the 3-sharp key signature,
  noteheads, beams, ties, fermatas, and accidentals.

## Decision

Render with **Verovio → per-page SVG → `rsvg-convert` (subprocess) → per-page PDF →
pypdf merge**, all hidden behind `Renderer`. Drop `cairosvg`.

`rsvg-convert` becomes a documented **system** dependency (`brew install librsvg`, or the
platform's librsvg package). `VerovioRenderer` checks for it via `shutil.which` and
raises `RenderError` with the install hint when it is missing.

## Consequences

- Not pure-Python: the renderer shells out. Acceptable — it is isolated in one class
  behind the interface, and the subprocess boundary is trivial (stdin SVG, stdout PDF).
- A future `LilyPondRenderer` implements the same `to_pdf` and needs no caller changes.
- If we later want zero system dependencies, the escape hatch is bundling a headless
  renderer or revisiting cairosvg with a controlled library path — but only if packaging
  forces it. For now, robustness beats purity.
- CI and contributor setup must install librsvg. `test_render.py` skips when
  `rsvg-convert` is absent so the rest of the suite stays runnable anywhere.
