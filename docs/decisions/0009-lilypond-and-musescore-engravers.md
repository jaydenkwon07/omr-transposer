# 0009 — Wiring the LilyPond and MuseScore engravers

Status: accepted (Project 1, 2026-07-15)

## Context

Verovio was the only live engraver; LilyPond and MuseScore were stubs (`available()` →
`False`). That made every generated pair a Verovio monoculture — the exact failure mode
CLAUDE.md's engraver thesis warns against ("three engravers beat a thousand augmentations of
one"). A model trained on one engraver learns that engraver and reads a real photo far worse
than its synthetic score suggests. Both remaining backends are external binaries with their
own quirks; this ADR records how each was tamed to fit the `Engraver` protocol.

## Decision

- **MuseScore: judge success by output, not exit code.** MuseScore 4's CLI
  (`mscore -r <dpi> -o out.png in.musicxml`) intermittently aborts during shutdown teardown
  (SIGABRT / rc=134, "mutex lock failed: Invalid argument") *after* the PNG is written and
  flushed. The output is valid and byte-identical to a clean run. Trusting `check=True`
  would skip ~1/3 of MuseScore samples, and since a skipped unit is redrawn, that silently
  breaks the `(corpus_id, seed, config_hash)` reproducibility contract. So we ignore the
  return code and look for the page PNGs; only a genuinely empty render is a failure (retried
  a few times in case the abort fired before the write).

- **LilyPond: two binaries, `musicxml2ly` → `lilypond --png`.** `musicxml2ly` (ships with
  LilyPond) converts MusicXML to `.ly`; `lilypond -dresolution=<dpi> --png` rasterizes it.
  `available()` requires both on PATH. Output is byte-deterministic and fast (~0.7 s/page vs
  MuseScore's ~2.4 s).

- **Suppress LilyPond's tagline footer** (`\paper { tagline = ##f }`, appended to the
  generated `.ly`). The default "LilyPond vX.Y.Z" footer is an engraver watermark with no
  counterpart in the MusicXML label — a leak that lets a model identify the engraver from
  its footprint instead of learning engraving-robustness. The score's own title/composer
  metadata is kept: it is identical across engravers and is legitimately part of the page.

- **Trim full-page margins to the ink box** (`trim_white_margins`, shared). MuseScore and
  LilyPond frame music inside full printer-page margins; Verovio emits tight SVG. Trimming to
  the ink bounding box (plus a small pad) makes the three engravers' framing comparable and
  keeps the ink fraction meaningful. Augmentation reintroduces crop/margin variance
  downstream, labelled and mask-safe (ADR 0007).

- **Composite transparent backgrounds over white** (`read_gray_over_white`, shared).
  MuseScore exports RGBA with a transparent background — the same trap CLAUDE.md flags for
  Verovio SVG. Compositing over white before grayscale makes paper paper regardless of what
  RGB sits under `alpha=0`, rather than relying on it happening to be white.

## Consequences

- All three engravers go live; the 3-engraver coverage test is green, not xfail. Over a
  large corpus the per-unit engraver binding (`_derive_seed(corpus_id, …)`, ADR 0007)
  distributes ~evenly across the three.
- Engraver is bound to the unit: a given musical unit is always rendered by the same
  engraver. Diversity comes from corpus breadth, not from re-rendering one unit three ways.
  Left as-is; revisit only if per-unit multi-engraver sampling proves worth the duplication.
- New runtime dependency on external binaries (LilyPond, MuseScore 4) for those engravers.
  Absent binaries drop out cleanly via `available()`; a machine with only Verovio still runs,
  just without the diversity. Tests that exercise MuseScore/LilyPond skip when the binary is
  missing (CI without them stays green but doesn't prove the render path).
