# 0013 — A trailing whitespace margin fixes CTC tail-token deletion

Status: accepted (Project 2, 2026-08-26)

Refines [0012](0012-ctc-gradient-clipping-and-canary-gate.md).

## Context

With gradient clipping in place (ADR 0012), the overfit-8 canary stopped failing 0/8, but a
fresh GPU run still **missed the token-SER ≤ 0.02 gate**, plateauing at **SER 0.076**. ADR
0012 had predicted a clean pass and attributed its earlier CPU plateau to the two examples
with **adjacent-duplicate** labels (a known greedy-decode edge). That explanation does not
survive the data below — the real floor is a different, larger effect.

## Investigation

A diagnostic (`scripts/canary_diag.py`) logged, every 100 epochs, train-mode loss vs
eval-mode loss (same batch) vs eval token-SER, then dumped per-example gold-vs-pred opcodes.

- **Not a BatchNorm gap.** train_loss and eval_loss track each other the whole way and end
  identical (0.0006 / 0.0006). The eval/predict forward reproduces training's.
- **Not undertraining.** Loss is driven to 0.0006–0.007; example 6 decodes exactly. The
  architecture *can* memorize. SER nonetheless froze at ~0.076 from epoch 1200→2500 while loss
  kept falling.
- **Not the adjacent-duplicate floor.** Across the 8 incipits only 2 of 171 tokens are
  adjacent duplicates (1.2%). The observed 0.076 is ~12 ops.
- **The error signature is decisive:** every residual error is a **deletion**, and they cluster
  at the **tail** of each sequence (e.g. gold[15:17], gold[21:23], the final `barline` /
  `rest-quarter`). This is not a duplicate artifact — distinct terminal tokens are dropped.

Measuring the images explained why: PrIMuS incipit PNGs are cropped **flush to the last
symbol** (0–1 trailing white columns; ink reaches the final pixel). CTC greedy needs at least
one trailing blank frame to commit the final label; with none, it emits blank there and drops
the tail. Near-zero *marginal* loss (which sums over all alignments) cannot fix a *best-path*
decode that has no frame to place the last symbol.

## Hypothesis test

Append white columns on the right of every preprocessed image (`OMRT_TRAIL_PAD`), applied in
`preprocess` so the **training** path (`PrimusDataset`) and the **eval** path
(`CRNNModel.predict`) share one frame budget. A/B on GPU, identical seed:

| | 0 pad | 64 px (16 frames) |
|---|---|---|
| token-SER floor | 0.076 (never passes) | **0.018 — reaches ≤0.02 at ep ~1300** |
| ≤0.02 first hit | never | ~ep 600 |
| tail deletions | 12 ops, all deletions | gone (5/8 exact) |

The tail deletions vanished and convergence got *faster*. The 3 residual errors became a
single phantom `note-D4_quarter` emitted in the long blank tail — a benign overfit-8 artifact
(same token every time; real-data volume regularizes it away), well under the gate.

## Root cause

Flush-cropped inputs leave CTC greedy no trailing blank frame to commit the final label →
systematic tail-token deletion. This would have inflated SER on the *real* training run too,
not just the canary; it is a genuine pipeline defect, not a test-threshold problem.

## Decision

- **Bake a right-side white margin into `preprocess`** — `_TRAIL_PAD = 64` post-resize columns
  (16 frames), overridable via `OMRT_TRAIL_PAD` for ablation. It is applied in the one function
  both seams share, so train and eval never disagree on the frame budget.
- Keep the canary gate at **token-SER ≤ 0.02** (ADR 0012); it is now genuinely reachable
  (~0.018) rather than blocked by an input defect.
- **Supersede 0012's adjacent-duplicate explanation** of the plateau. Adjacent-duplicate
  collapse is real but tiny (1.2% here); the dominant floor was tail-deletion, now removed.

## Consequences

- The mandated overfit-8 canary passes honestly with padding as the default (no env var).
- The width constraint `T = W/4 ≥ L` only gains slack (padding widens W), so the collate-time
  assertion is unaffected.
- Not changed: the CRNN architecture, `decode`, the vocab, the metric, or the gradient-clipping
  fix from 0012.
- Leading-edge padding was *not* added — no leading errors were observed. Revisit only if real
  data shows head-token deletion.
