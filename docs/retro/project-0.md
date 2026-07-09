# Project 0 retro — the symbolic layer

*Draft. Status: awaiting review.*

## What shipped

`src/omrt/symbolic/`, a complete CLI that transposes MusicXML and renders a PDF:

- `transpose.py` — three modes over MusicXML strings (str → str, no I/O):
  `transpose_to_key`, `transpose_by_interval`, `transpose_for_instrument`.
- `keys.py` — key detection (explicit notation before analysis) and key-name parsing.
- `spelling.py` — `normalize_spelling`, dependency-free so Project 3 can reuse it as a
  label canonicalizer.
- `instruments.py` — transposing-instrument registry with explicit transpositions.
- `render.py` — `Renderer` protocol + `VerovioRenderer` (Verovio → SVG → rsvg-convert →
  pypdf merge).
- `cli.py` — `omrt transpose`, the only module that touches files.

Quality gates: `mypy --strict` clean on `src/`; 37 tests green, using hypothesis for the
music-theoretic invariants (round-trip, interval preservation, key-signature match,
no-spurious-doubles).

Three ADRs recorded the decisions the project was expected to surface, plus one extra:
0001 renderer, 0002 enharmonic spelling, 0003 key detection.

## What got thrown away

Almost nothing — this is the "survives unchanged" project. `cairosvg` was dropped from
the dependency list after the spike (see below). The rest is the asset.

## What surprised me

- **`instrument.Trumpet()` already defaults to a Bb transposition (M-2)** in music21 9.x,
  not the C/P1 we expected. The lesson held anyway: set `transposition` explicitly, since
  defaults have drifted across versions.
- **cairosvg was the weak link, but not for the reason we guessed.** We expected trouble
  with SMuFL `<use>`/`<symbol>` glyph rendering. Instead it would not even import:
  `cairocffi` can't find Homebrew's `libcairo` by leaf name, and the fix (`DYLD_*`) has to
  be set before process start — impossible for a console-script CLI to guarantee.
  `rsvg-convert` sidesteps all of it as a self-contained binary. → ADR 0001.
- **Getting a MusicXML *string* out of music21 is not obvious.**
  `ConverterMusicXML().write(..., subformats=["str"])` returns a `PosixPath` (it writes a
  temp file). The string path is `GeneralObjectExporter(score).parse().decode()`.
- **`toWrittenPitch` transposes the key signature too**, and the round-trip through
  MusicXML export is safe *only* because we leave the score at `atSoundingPitch = False`;
  otherwise the exporter's own `toWrittenPitch` double-transposes. The test that
  re-parses the export and asserts exactly one M2 is what pins this down.
- **music21 is only partially typed**, so `mypy --strict` needed a handful of narrow
  `cast`s and two localized `type: ignore`s — not a blanket relaxation.

## Numbers

- 37 tests, ~3–4 s wall.
- `mypy --strict`: 8 source files, no issues.
- Rendered a 2-system Bach chorale to a ~66 KB PDF per mode; glyphs verified by eye.

## Open questions carried forward

- Mode inference when only a plain `KeySignature` is present (currently assume major +
  warn; ADR 0003).
- Modulating scores: `_force_key_signature` sets every key signature to the target, so a
  score that modulates loses its relative modulation in the notated signatures (notes are
  still transposed by one interval). Documented as an xfail in the trap suite; deciding
  the right behavior belongs in ADR 0002's scope section.
- Two trap-suite xfails mark scope questions ADR 0002 has actually now *decided*
  differently than the tests assume (theoretical target key: accept + warn, not reject).
  Those xfails should be converted to real assertions of the decided behavior, or the
  decision revisited.

## Resolved during the trap suite

- **Register / "short way" in to-key transposition.** Previously we used the raw
  tonic-to-tonic interval, which sent C major → B major up a major 7th. Now
  `_shortest_transposition` picks the nearest octave (ties resolve upward, keeping C → F#
  an ascending A4), so C → B descends a minor 2nd. This was the only real bug the trap
  suite found; everything else (accidental display status, chord-symbol figures, Bb
  written-part idempotency) already held.
