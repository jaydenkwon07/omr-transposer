# 0011 — The semantic decode path and its round-trip ceiling

Status: accepted (Project 2, 2026-08-13)

## Context

ADR 0010 rejected the MEI route. Seam 3 decodes PrIMuS `.semantic` straight to MusicXML.
The round-trip ceiling — `to_symbols(decode(tokens))` vs `tokens` over the real corpus —
bounds model accuracy and must be measured before training. A CRNN cannot score better than
the decode path can round-trip; whatever the decoder loses is subtracted from the best
achievable SER before a single weight is learned.

## Decision

`decode` builds music21 objects (the inverse of `eval/symbols.py::to_symbols`) and exports
with `makeNotation` disabled. Measured over the 1,885-incipit sample
(`data/primus_sample/package_aa/`, gitignored) via `tests/test_decode_ceiling.py`:

- **RAW ceiling SER = 1.7744%**, token-identical on 1,072/1,885 incipits.
- **NORMALIZED ceiling SER = 0.0045%**, exact on **1,883/1,885** incipits.

The test hard-asserts NORMALIZED SER < 0.5%. RAW is reported for this record.

## The trailing-barline convention artifact

`to_symbols` emits a `barline` after every measure, but **813/1,885 incipits (43%) omit
their final barline**. A perfect decoder therefore diverges by one trailing `barline` on
those, and that single class accounts for essentially all of the RAW SER (813 raw mismatches,
811 of them barline-only). This is a metric tokenization convention, not lost musical
information, and it CANCELS in real model evaluation — there both prediction and truth pass
through `decode → to_symbols`, so the appended barline appears on both sides. The NORMALIZED
number strips a single trailing `barline` from both sides and is the true decode-fidelity
ceiling.

## Documented losses (normalized)

Two incipits remain, both the same irreducible class: a **redundant mid-stream attribute
restatement that music21 collapses on export.**

| incipit | dropped token | why |
|---|---|---|
| `000103512-1_1_2` | `timeSignature-3/4` restated at a measure boundary where the meter is unchanged | music21 does not re-emit an attribute equal to the one already in force |
| `000141774-1_1_2` | `clef-C1` restated mid-measure, identical to the active clef | same — a no-op clef change is not serialized |

Both are genuine export losses, not decode bugs: the decoder faithfully creates the object,
but a restatement that changes nothing is dropped on the MusicXML round trip. Neither carries
musical information (the meter/clef is already what the restatement asserts), so neither is
worth defeating `makeNotation=False` to preserve. Counted here so the 0.5% budget is spent on
enumerated losses, never on a hidden defect.

One decode gap *was* found and fixed while measuring this ceiling: the `longa` /
`quadruple_whole` duration was absent from the `_DURATION_TYPES` / `_DURATION_NAMES` inverse
pair. It is now mapped in both directions (the inverse-dict test guards the pair).

## Consequences

- The decode path is the sole tokens→MusicXML route for Project 2 (ADR 0010). The NORMALIZED
  ceiling (0.0045%) is the accuracy the CRNN cannot exceed on this vocabulary; the model's job
  is to close the gap to it, not to it-plus-a-metric-artifact.
- The RAW/normalized gap (~1.77 points) is a metric convention, tracked here so no one mistakes
  it for a decode defect and chases it.
- `tests/test_decode_ceiling.py` re-measures this on every run (skips if the sample is absent);
  a regression that pushes NORMALIZED SER over 0.5% fails the suite with the offending incipits
  printed.
