# 0003 — Key detection: explicit notation first, analysis as a last resort

Status: accepted (Project 0, 2026-07-09)

## Context

`transpose_to_key` needs the score's *source* key to compute the interval to the target
tonic. There are two ways to get it, and they are not equally trustworthy:

- **Read the notated key signature.** Deterministic, and it is what the engraver wrote.
- **`score.analyze('key')`.** A Krumhansl-style heuristic over the pitch content. It is a
  guess; it can be wrong, especially on short excerpts, modal music, or heavy chromaticism.

Using analysis when an explicit key signature is present would be throwing away ground
truth in favor of a guess. But a key signature alone is ambiguous about *mode* (2 sharps
is D major or B minor), and a plain `KeySignature` element carries no mode.

## Decision

`keys.detect_key(score)` resolves in strict precedence:

1. An explicit `music21.key.Key` element (carries tonic **and** mode) — used outright, no
   warning. This is the common case for scores exported with a real key.
2. Otherwise, a plain `music21.key.KeySignature` (sharp/flat count, no mode) — assume
   **major**, and emit a `KeyDetectionWarning`. The sharp count is ground truth; the mode
   is an assumption the user should see.
3. Otherwise (no key signature at all) — fall back to `score.analyze('key')` and emit a
   `KeyDetectionWarning` flagging that the result is heuristic.

## Consequences

- Determinism when the notation is unambiguous; a visible warning whenever we assume or
  guess. Warnings are `KeyDetectionWarning` (a `UserWarning` subclass) so callers can
  filter or escalate them.
- Mode assumption in case (2) can be wrong for minor-key pieces notated with only a
  KeySignature. Acceptable for Project 0; a future improvement could infer mode from the
  final bass note or tonic emphasis, but that is itself heuristic and out of scope now.
- The transposition interval is computed between tonic *pitches* regardless of which
  branch produced the key, so spelling correctness (ADR 0002) does not depend on
  detection method.
