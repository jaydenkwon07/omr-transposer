# omr-transposer

Photo of sheet music in, transposed notation out. See `CLAUDE.md` for the full framing.

**Project 0 (current): the symbolic layer.** MusicXML in → transposed MusicXML
(music21) → PDF (Verovio). Transposition and rendering are solved, deterministic
problems; the OMR that produces the MusicXML comes in later projects.

## Setup

```bash
uv sync
brew install librsvg      # provides rsvg-convert, used for SVG -> PDF
```

## Usage

```bash
omrt transpose score.musicxml --to-key "F# major"     --out out.pdf
omrt transpose score.musicxml --by-interval m-3       --out out.pdf   # descending: m-3 or -m3
omrt transpose score.musicxml --for-instrument bb-trumpet --out out.pdf
```

Instruments: `bb-trumpet`, `bb-clarinet`, `a-clarinet`, `eb-alto-sax`, `bb-tenor-sax`,
`f-horn`.

## Develop

```bash
uv run pytest        # 37 tests; hypothesis for the music-theoretic invariants
uv run mypy          # strict, on src/
```

Design decisions live in `docs/decisions/` (ADRs).
