# 0010 — Bypass the PrIMuS MEI bridge; reach MusicXML through seam 3

Status: accepted (Project 2, 2026-07-27)

## Context

PrIMuS ships each incipit five ways, two of which can serve as ground truth: `.mei` and
`.semantic`. The Stage A design assumed the MEI route — `music21` registers a `ConverterMEI`,
so `.mei` → music21 → MusicXML falls out with no code of ours in the middle — and a spike
recorded it as verified on the strength of 300/300 parse, export, and **note-count** match.

Calibrating the finished seam-4 metric against the shipped `.semantic` contradicted that.
Over 300 randomly sampled incipits (`scripts/calibrate_primus.py --sample 300 --seed 0`):
**18.79% SER, 5.0% exact match, 85.2% token overlap.** The aligned diff attributes nearly all
of it to music21's MEI parser rather than to our grammar:

| cause | divergent ops | behavior |
|---|---|---|
| key-signature accidentals ignored | 765 of ~1,351 | `<note pname="b"/>` under `key.sig="3f"` parses as B natural; PrIMuS says `Bb` |
| `meter.sym` dropped | ~101 | no `TimeSignature` object at all, so every `timeSignature-C` / `C/` vanishes — and those are >50% of the corpus's time signatures |
| `<multiRest>` dropped | 92 + knock-on rest/barline deltas | later backfilled as an ordinary rest |

Per-class count divergence is 0 for clef, keySignature and gracenote, which is what says the
grammar itself is sound.

The count-based spike check could not have caught this: every note *is* present, so counts
match exactly while ~90% of the accidentals are gone.

## Decision

**Do not use the MEI route for PrIMuS ground truth. Do not repair it either.**

PrIMuS's `.semantic` files are already the token vocabulary Project 2 trains on. Seam 3 —
`decode(tokens) -> MusicXMLStr`, built in Stage B — converts them to MusicXML directly. That
is the training-label path regardless, so the MEI bridge was always a second, parallel route
to the same destination, and it is the lossy one.

The planned Stage B round-trip ceiling test measures exactly this path (tokens → MusicXML →
tokens, over the whole corpus), so the calibration Stage A could not honestly produce is not
lost — it moves to where it is meaningful.

`data/primus_sample/package_aa/` keeps its `.mei` files; nothing needs deleting. They are
simply not an input to the pipeline.

## Consequences

- Stage A's calibration number (18.8%) is **not** a standing offset on our metric and must not
  be quoted as one. It is a measurement of a route we no longer take. `scripts/calibrate_primus.py`
  stays as the record of why.
- Stage B gains a hard requirement it did not have: seam 3 must decode the full `.semantic`
  grammar, because there is no fallback route to MusicXML. The token census in the Stage A spec
  is the checklist.
- The four defects above are music21 upstream behavior on PrIMuS's MEI subset. If a future
  project wants MEI as an independent cross-check, the fix list is here — but nothing in
  Projects 2–3 needs it.
- Anything else that reaches for `ConverterMEI` should assume the same gaps until measured.

## Alternative rejected

**Repair the bridge**: a PrIMuS loader applying key-signature accidentals to unaltered pitches
and reading `meter.sym` / `<multiRest>` from the raw MEI. Roughly half a day, and it buys an
independent check on seam 3 that nothing currently requires. Repairing a dependency we can
delete instead is the worse trade.
