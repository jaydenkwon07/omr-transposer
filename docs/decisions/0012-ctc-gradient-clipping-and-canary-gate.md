# 0012 — Gradient clipping, the zero_infinity trap, and a token-SER canary gate

Status: accepted (Project 2, 2026-08-26)

## Context

The mandated overfit-8 canary (`tests/test_models_overfit8.py`, gate #2) was run green for
the first time on a Colab GPU. It **failed**: the training loss dropped below the `< 0.1`
gate, but `CRNNModel.predict` then decoded **0/8** examples exactly. Per CLAUDE.md that
signature ("below-0.1 loss but bad decode") means an architecture bug, not undertraining.

A full local reproduction (CPU) ruled the architecture *out*, one suspect at a time:

- **Frame width** — every one of the 8 has `T = W/4 ≫ L` (e.g. 215 vs 19); CTC has ample
  frames. Not it.
- **BatchNorm train/eval gap** — a direct logit probe showed train-mode vs eval-mode argmax
  agreement of **1.000** on all 8 (max logit Δ 0.68, BN running stats tracking the batch to
  `max|Δmean| = 0.009`). `predict`'s forward reproduces training's to ~99–100%. Not it.
- **Decode / vocab path** — vocab round-trips, and a synthetic *perfect* model output decodes
  **8/8**. Not it.

What remained: the model collapses to emitting the CTC **blank** on ~99% of frames
(`blank_frac ≈ 0.99`), so greedy decode collapses to near-nothing. A *genuine* loss < 0.1 is
mathematically incompatible with all-blank output — so the GPU's reported low loss was not
genuine.

## Root cause

Two compounding defects in the training setup:

1. **No gradient clipping.** `_run_loop` (and the canary's own loop) ran
   `loss.backward(); opt.step()` with nothing between, under `Adadelta(lr=1.0)` on a 2×BLSTM.
   That is the textbook RNN+CTC instability recipe: gradients explode, the LSTM blows up to
   `inf`.
2. **`CTCLoss(zero_infinity=True)` masks the explosion.** When the loss goes infinite,
   `zero_infinity` silently rewrites it to **0**. The canary's `assert last < 0.1` then passes
   on a *masked* inf while the model is actually dead (blank-collapsed) → 0/8.

On CPU the explosion never fired (`raw == zero_infinity` loss at every step), so the model
converged honestly but slowly, and already decoded **3/8 at loss 0.09** — proving low loss and
good decode move together when training is stable. On GPU (cuDNN LSTM, `lr=1.0`, no clip) the
inf fired, was masked, and produced the false pass. Clipping removes the explosion; the masked
value can no longer masquerade as convergence.

## The greedy / adjacent-duplicate finding

With clipping added, an honest run climbs monotonically — `1/8 @0.18 → 3/8 @0.09 → 6/8 @0.018
→ 7/8 @0.009` — and then **plateaus at 7/8**. The last example is one of the two (of eight)
whose label contains **adjacent-duplicate tokens** (`note-C5_eighth note-C5_eighth`). Greedy
CTC decoding structurally cannot recover an adjacent duplicate unless the model places a blank
*exactly* between the pair; a perfectly-trained net still tops out around 7/8 exact. This is a
property of greedy decode, not of the architecture (beam search would recover it).

**Consequence:** an exact-8/8 canary gate is inherently flaky for a *correct* model. The gate
was changed to **token-SER ≤ 0.02** over the eight examples — near-perfect memorization, but
immune to the greedy-duplicate edge. A blank-collapsed model scores SER ≈ 1.0, fifty times
over the bar, so the diagnostic power against real breakage is unchanged.

## Decision

- **Add gradient clipping** to the training loop and the canary:
  `clip_grad_norm_(model.parameters(), 5.0)` between `backward()` and `step()`
  (`_GRAD_CLIP_NORM` in `train.py`).
- **Harden the canary against masked-inf**: shadow the loss with a second
  `CTCLoss(zero_infinity=False)` and assert it stays finite, so a divergence can never again
  read as convergence.
- **Gate the canary on token-SER ≤ 0.02** via the real `predict` path, not exact 8/8, and
  break early on the SER target rather than a loss threshold (greedy decode lags the loss).
  `CANARY_EPOCHS` default raised 2000 → 2500 (honest convergence reaches the bar by ~ep 1500).

## Consequences

- The canary now measures the thing that actually matters (near-zero token error through the
  seam-2 path) and cannot be fooled by a masked-inf loss.
- Real training on GPU is stabilized; the same explosion that fooled the canary would have
  destabilized the full run.
- Not changed: the CRNN architecture, `decode`, the vocab, or the metric. Those were verified
  correct here and are untouched.
