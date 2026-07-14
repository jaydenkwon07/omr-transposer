# 0007 — Crop safety: no ink leaves the frame (a deliberate v1 gap)

Status: accepted (Project 1, 2026-07-13)

## Context

The silent killer of a synthetic OMR set is a geometric transform that pushes part of a
system off the page while the label still claims it — image and label silently disagree and
the model is trained on a lie. Crop is the obvious culprit, but it is not the only one: a
rotation or perspective warp can push a corner of content out of frame on its own, and
checking the crop in isolation misses the crop × rotate × warp interaction. Checking
Verovio's SVG geometry would catch it for Verovio but break for LilyPond and MuseScore.

## Decision

**Image-space invariant, engraver-agnostic.** Before augmenting, compute an ink mask of the
rendered page. Push that mask through the *exact same composed geometric pipeline* as the
image (same sampled rotation, perspective, curl, crop). A dropped system is, by definition,
mask pixels that left the frame, so we assert none do — two guardrails in `augment`:

1. After the warp (on a padded canvas), no ink may touch the padded edge — catches a
   rotation/perspective that clipped content.
2. The crop rectangle must contain the whole ink bounding box — catches the crop itself.

Violations raise `CropSafetyError` rather than emitting a poisoned pair.

**Accepted consequence, on purpose:** "no ink ever leaves the frame" means the model never
sees partial-page photos — a cut-off last system, a finger over a measure — which are
extremely common in real phone shots. This is a real gap, not a bug. It is a fine v1
simplification because it keeps image and label consistent *for free*. The realistic
version crops *into* content and crops the label to match; that is a genuine feature (label
surgery, not an afternoon's fix) and is deferred deliberately.

## Consequences

- The invariant holds under any composition of transforms and any engraver.
- Augmentation ranges are kept conservative (±3° rotation, ≤3% perspective, generous warp
  pad) so the guardrails are a safety net, not a frequent rejection path.
- Training data is biased toward fully-framed pages. A future "partial page" augmentation
  must crop the label in lockstep and will supersede the no-ink-leaves-frame rule.
