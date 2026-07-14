# 0002 — Enharmonic spelling: keep diatonic doubles, respell spurious ones

Status: accepted (Project 0, 2026-07-09)
Note: the "Theoretical target keys" decision below is superseded by [0004](0004-reject-out-of-scope-to-key.md) (reject instead of accept-and-warn). The enharmonic-spelling decision stands.

## Context

Transposition must produce *correctly spelled* pitches, not just correct pitch classes.
music21's interval transposition already does this well: it preserves the spelled
interval, so C major → F# major sends B to E# (not F). But two spelling questions need a
committed, precise answer because Project 3 will reuse this exact normal form to
canonicalize model training labels:

1. When is a double accidental (𝄪 / 𝄫) *wrong*?
2. What does a *theoretical* target key (more than 7 sharps/flats, e.g. G# major) do?

The naive answer to (1) — "always simplify double accidentals" — is wrong. F𝄪 is the
correct leading tone of G# major; simplifying it to G would corrupt the notation.

## Decision

### Definition of a *spurious* double accidental

A pitch carrying a double accidental (`abs(accidental.alter) == 2`) is **spurious** iff
it is **not** diatonic to the governing key context — i.e. its spelled name is not one of
`key_context.pitches`. Spurious doubles are respelled to their simplest enharmonic
(music21 `simplifyEnharmonic(mostCommon=True)`), carrying the octave from the simplified
pitch so the sounding pitch never changes (B𝄪4 → C#5, not C#4). Diatonic doubles are
left untouched (F𝄪 in G# major stays F𝄪).

Single accidentals are never touched. With **no** key context (interval mode), spelling
is left exactly as music21 produced it: interval transposition is already canonical, and
there is no tonal frame in which "simpler" is well-defined.

This lives in `spelling.normalize_spelling(score, key_context)`, which has **no**
dependency on `transpose` — Project 3 imports it as a label canonicalizer.

### Theoretical target keys

`--to-key "G# major"` (8 sharps) is **accepted as requested** and the score is
transposed to it, keeping its diatonic double accidentals. We emit a warning naming the
practical enharmonic (`Ab major`). We do **not** silently respell to Ab (that would
violate the user's explicit request, the same way honoring E# over F is the whole point),
and we do **not** error (too restrictive; the key is representable and music21 handles
it).

## Consequences

- The normal form is "no double accidentals except those diatonic to the key" — precise,
  testable (`test_no_spurious_double_accidentals`, `test_spelling.py`), and reusable.
- Correctness depends on `key_context.pitches` listing diatonic doubles, which music21
  does (G# major → `[..., E#, F##, G#]`). Verified in Project 0.
- A user who wants Ab instead of G# passes `--to-key "Ab major"`. The warning points the
  way.
