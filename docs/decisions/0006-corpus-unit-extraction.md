# 0006 — Corpus unit extraction: voice single-staff + piano grand-staff

Status: accepted (Project 1, 2026-07-13)

## Context

The starting corpus is OpenScore Lieder. A Lieder score is voice + piano — measured as ~3
parts (a vocal `Part` plus two `PartStaff` forming the piano grand staff), occasionally
more for duets/choral pieces. The Project 1 non-goal is "single-staff and piano grand staff
only". Taken as whole pages, almost the entire corpus exceeds that ceiling, so a naive
staff-count filter would discard nearly everything.

But the non-goal describes the *unit* we generate, not the whole page, and music21 tags the
structure for us: a plain `Part` is a single staff, a `PartStaff` is one staff of a
keyboard group. The corpus therefore decomposes cleanly into exactly the two permitted unit
types.

## Decision

The corpus loader extracts **units**, not whole scores:

- Each plain vocal `Part` → one **single-staff** unit.
- Each contiguous run of `PartStaff` → one **grand-staff** unit (with a braced `StaffGroup`
  so an engraver draws the two staves as one keyboard system). A run longer than
  `max_staves` is dropped, not truncated.

Each unit is materialized as a fresh `Score` (deep-copied parts) so rendering and labelling
never see the rest of the page. `corpus_id` is `"<relative-path>#<tag>"` (e.g.
`Composer/Song/lc123.mxl#s0`, `...#g0`), stable across runs.

Considered and rejected: (a) single-staff-only, which splits the piano into musically-odd
half-parts and drops grand-staff coverage; (b) whole 3-staff page, which is a
multi-instrument page and violates the non-goal without a human lifting it.

## Consequences

- The whole Lieder corpus is usable while staying inside the non-goal.
- One score yields multiple training units of both types — good for coverage.
- Units are decontextualized: a vocal line loses its piano accompaniment on the page. That
  is correct for single-staff/grand-staff OMR and matches the project's current ceiling.
- When multi-instrument pages become a goal (a later project), whole-page units supersede
  this — they do not modify it.
