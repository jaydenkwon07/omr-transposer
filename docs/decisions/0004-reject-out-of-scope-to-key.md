# 0004 — Reject theoretical target keys and modulating inputs in to-key mode

Status: accepted (Project 0, 2026-07-13)
Supersedes: the "Theoretical target keys" decision in [0002](0002-enharmonic-spelling.md).

## Context

`transpose_to_key` computes a single directed interval from the source tonic to the
target tonic and applies it uniformly to the whole score. Two inputs have no well-defined
answer under that model, and both were previously left undecided (tracked as `xfail`s in
`test_traps.py`, TRAP 9):

1. **Theoretical target keys** — a target with more than 7 accidentals (e.g. G# major,
   8 sharps). ADR 0002 chose to *accept and warn*, transposing to the key and emitting a
   warning that named the practical enharmonic (Ab major).
2. **Modulating inputs** — a score whose notated key changes partway through (C major then
   G major). A single interval honors the target for the first key and wrongly shifts
   every later section.

Revisiting (1): accept-and-warn produces a page that needs a double-sharp/-flat key
signature, which no conventional reader or downstream renderer expects, and which Project
3 would then have to treat as a valid training label. The warning is easy to miss and the
output is a footgun. The cost of being strict is one clear error message; the cost of
being lax is silently wrong notation.

## Decision

Both cases **raise `ValueError`** from `transpose_to_key`.

- **Theoretical key** (`abs(target.sharps) > 7`): reject, naming the practical enharmonic
  to request instead (`_reject_if_theoretical`). A user who wants Ab passes `Ab major`.
- **Modulation** (more than one distinct key-signature sharp count in the score):
  reject, pointing at `transpose_by_interval` for a uniform shift that ignores key
  (`_reject_if_modulating`). Distinct *sharp counts* — not mode changes — define
  modulation here, because the transposition interval is driven by the notated signature,
  and a relative major/minor change shares one signature.

Both live in `transpose.py` and guard `transpose_to_key` only; interval and instrument
modes are unaffected.

## Consequences

- The two TRAP 9 `xfail`s become ordinary passing tests asserting the raise.
- `test_theoretical_key_is_accepted_with_warning` is replaced by
  `test_theoretical_key_is_rejected`; the theoretical-key warning no longer exists.
- to-key mode now has a crisp contract: one source key in, one notatable target key out,
  or a `ValueError` that says what to do instead. This is the normal form Project 3 will
  canonicalize training labels against, so narrowing it is a feature, not a limitation.
- If multi-key transposition is ever wanted, it is a deliberate new feature (per-section
  interval computation), not a silent fallback — and would supersede this ADR.
