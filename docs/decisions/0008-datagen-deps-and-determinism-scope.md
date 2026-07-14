# 0008 — Datagen deps: headless OpenCV, optional group, same-machine determinism

Status: accepted (Project 1, 2026-07-13)

## Context

The raster pipeline needs numpy, Pillow, and OpenCV. Three choices came with that: which
OpenCV build, where the deps live, and what "reproducible" actually guarantees.

## Decision

- **`opencv-python-headless`, not `opencv-python`.** The pipeline never opens a window;
  the GUI build pulls in highgui/Qt that cause grief on headless boxes and in CI for no
  benefit. Same `cv2` API (`warpPerspective`, `remap`, JPEG encode).
- **Optional dependency group `[project.optional-dependencies] datagen`**, not core. The
  Project 0 CLI transposes and renders PDFs without touching cv2; it stays installable
  without the heavy raster stack. Install with `uv sync --extra datagen`.
- **OpenCV is version-pinned** (`==4.10.0.84`). `warpPerspective`/`remap`/JPEG encoding are
  only byte-reproducible within one OpenCV build.
- **Determinism is scoped same-machine.** "Same seed → byte-identical images" holds within
  one environment; it is *not* promised across OpenCV versions or platforms. The
  determinism test asserts within-process reproduction, so it does not go flaky the first
  time CI runs on a different OS than the author's laptop.
- **mypy treats `cv2` as opaque** (`follow_imports = "skip"`); its bundled stubs fight
  strict mode over trivia. Typing is restored where a cv2 result is returned (`np.asarray`).

The connective tissue: one `np.random.Generator` is threaded through both engraver
selection and every augmentation transform. No transform reaches for global `np.random`, so
the per-sample seed fully determines the output.

## Consequences

- Core install stays lean; raster stack is one extra away.
- Reproducibility claims are honest about their boundary (same machine), which is what the
  `(corpus_id, seed, config_hash)` triple needs to answer "did the model get worse, or the
  data?".
- Bumping OpenCV is a deliberate act that may change pixels; the pin makes that visible.
