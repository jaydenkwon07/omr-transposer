# CLAUDE.md

Project context for `omr-transposer`. Read this fully before acting.

---

## What we are building

An application that accepts an image of printed sheet music and outputs the same
music transposed into a user-chosen key, rendered as readable notation.

## The single most important framing

**Transposition is not an AI problem.** Once music exists as structured symbolic data
(MusicXML), transposing is deterministic interval arithmetic. `music21` does it correctly,
including enharmonic spelling, key signatures, and transposing-instrument conventions.

**Rendering is not an AI problem.** Verovio and LilyPond do it.

The *entire* difficulty of this product lives in one stage: turning a photo into symbolic
music. That field is Optical Music Recognition (OMR). All model work targets that stage
and only that stage.

```
photo ──▶ [ OMR model ] ──▶ MusicXML ──▶ [ music21 ] ──▶ [ Verovio ] ──▶ output
             ^^^^^^^^^                     transpose        render
             the hard part                 solved           solved
```

## Ground truth about the state of the art

Do not over-promise accuracy. These are published, measured numbers as of mid-2026:

| System | Task | Result |
|---|---|---|
| CRNN + CTC (Calvo-Zaragoza & Rizo 2018) | single-staff monophonic, clean | ~2% symbol error rate |
| Transcoda (2026) | synthetic rendered pages | 18.46% OMR-NED |
| Legato (2025) | synthetic rendered pages | 43.91% OMR-NED |
| Transcoda | real historical scans | 63.97% OMR-NED |
| SMT++ (2024) | real historical scans | 80.16% OMR-NED |

Read that last column again. On real scanned pages, the best open system still gets roughly
two-thirds of the transcription wrong by normalized edit distance. Clean synthetic pages are
far better; photographs of real paper are the hard case.

**Product consequence:** every serious OMR tool ships a correction editor. Audiveris is built
around the tight integration of an OMR engine *and* an OMR editor. The editor is not an
admission of failure; it is the product. Design for human correction from the start.

**Research consequence:** the dominant lever on accuracy is *training data quality*, not model
scale. Transcoda trained a 59M-parameter model in six hours on one GPU and beat
billion-parameter baselines, by investing in synthetic data generation, label normalization,
and grammar-constrained decoding. We follow that philosophy.

---

## Architecture: four seams

These interfaces are fixed. Every future component plugs into them. Do not violate them
for convenience.

```python
# 1. Data generation. Ground truth is ALWAYS MusicXML, never a model-specific format.
def generate(n: int, config: GenConfig) -> Iterator[tuple[Image, MusicXMLStr]]: ...

# 2. The model socket. Every model — CRNN, transformer, fine-tuned checkpoint — hides here.
class Model(Protocol):
    def predict(self, image: Image) -> list[Token]: ...

# 3. Format absorption. One decoder per output vocabulary (**kern, ABC, LMX, semantic).
def decode(tokens: list[Token]) -> MusicXMLStr: ...

# 4. Everything meets here. All comparison happens in MusicXML space.
def evaluate(predicted: MusicXMLStr, truth: MusicXMLStr) -> Metrics: ...
```

Rationale for seam 3: competing systems emit different formats — SMT++ emits `**kern`,
Legato emits ABC. MusicXML is the unifying format for evaluation. Converting at the decoder
boundary means we can benchmark any model against any other with the same `evaluate()`.

## Repo layout

```
omr-transposer/
├── CLAUDE.md
├── pyproject.toml
├── src/omrt/
│   ├── symbolic/     # Project 0. music21 wrapper, transposition, rendering.
│   ├── datagen/      # Project 1. MusicXML corpus -> (image, label) pairs + augmentation.
│   ├── models/       # Projects 2-5. Each implements Model.predict().
│   ├── decode/       # tokens -> MusicXML, one module per vocabulary.
│   └── eval/         # SER, OMR-NED, TEDn.
├── tests/
└── data/             # gitignored. corpora, generated sets, checkpoints.
```

---

## Roadmap

Each project is a complete working thing. **Models are scaffolding; the pipeline is the asset.**
Roughly 30% of code written once and kept, 30% reusable scaffolding, 40% thrown away.
This is the normal and correct ratio.

| # | Build | Teaches | Survives to production? |
|---|---|---|---|
| **0** | MusicXML in, transposed PDF out | symbolic music data structures | **Yes, unchanged** |
| 1 | Synthetic data generator + augmentation | where accuracy actually comes from | **Yes, compounds** |
| 2 | CRNN + CTC on PrIMuS, single staff | image→sequence, CTC, SER | No — CTC can't express chords |
| 3 | Encoder-decoder transformer on grand staff | tokenization, label normalization, beam search | Partially (tokenizer, normalizer) |
| 4 | Fine-tune SMT++ / Legato checkpoint | transfer learning, frozen encoders, humility | Maybe |
| 5 | Full page, multi-instrument, curriculum learning | the research frontier | This is the endgame |

Evaluation harness and the hand-annotated real-photo test set are built early and outlive
every model. They are the most valuable artifacts in the repo.

---

## Project 1 — data generator (built; carried debt below)

Durable facts, kept because regenerating without them repeats old mistakes:

- Labels are MusicXML, normalized via `spelling.normalize_spelling()`. Never a token vocabulary.
- Every pair reproducible from `(corpus_id, seed, config_hash)`. This is what answers
  "did the model get worse, or the data?"
- Variance priority: **engraver** > photometric > geometric > corpus diversity. Three engravers
  beat a thousand augmentations of one — it is the only reason this generator beats
  Camera-PrIMuS, which is itself Verovio-with-3-fonts.
- Verovio SVG has a transparent background. Rasterize with a white background or the image
  collapses to solid black in grayscale.
- Determinism holds *within one environment only* — cv2 warp/JPEG differ across versions.

---

## Project 2 — CRNN + CTC on single staves (ACTIVE)

Three seams wake up at once. Projects 0–1 only exercised seam 1.

**Build order is not negotiable: seam 4, then seam 3, then the model.**
A buggy metric reports improvement that isn't there, and you will chase it for weeks.

1. **Seam 4 first** — `evaluate()`. Symbol error rate. Degenerate tests: `evaluate(x,x) == 0`
   exactly, `evaluate(empty,x) == 1` exactly, symmetry if claimed.
2. **Seam 3 second** — `decode(tokens) -> MusicXML`, and its inverse for training labels.
   Then the **round-trip ceiling test**: encode every PrIMuS score to tokens, decode back,
   assert semantic equivalence. The failure count across the corpus IS the model's accuracy
   ceiling. Know it before training anything.
3. **Seam 2 last** — the CRNN behind `Model.predict(image) -> list[Token]`.

**Vocabulary: semantic first, agnostic as an ablation.** PrIMuS ships both. Semantic carries
musical meaning (a D major key signature is one symbol); agnostic is position-only graphics
(the same key signature is two "sharp" symbols) and needs a real parser in seam 3. Semantic
closes the loop fastest, which is what validates the seam chain. The agnostic ablation is the
Project 3 tokenization lesson, one project early and cheap once the harness exists.

**Seam 4 bridge:** PrIMuS ships MEI; `music21` registers a `ConverterMEI` for `.mei`; MusicXML
falls out. Verify on a real PrIMuS file — registration does not guarantee their MEI subset parses.

**Data plan, three stages:**

1. PrIMuS clean → target ~2% SER. The only project with a published number. Missing it means
   the implementation is wrong, not the task.
2. Camera-PrIMuS (same incipits, photo-distorted) → the delta is what noise costs.
3. Our own Project 1 vocal single-staff units → the first real report card on the generator.
   **Blocked until the two Project 1 bugs are fixed.**

**Two CTC-specific failure modes:**

- **Width constraint.** CTC needs output frames ≥ label length. Pool too hard in width and dense
  incipits become mathematically unemittable → inf/NaN loss. Assert the ratio explicitly.
- **Overfit one batch.** 8 examples, augmentation off, dropout off, train to near-zero loss.
  ~90 seconds. Separates "architecture is broken" from "data is broken" — otherwise a week of
  guessing. Write this before the training loop.

**CTC's wall is the point.** It assumes one symbol per frame, monotonic left-to-right. It
*cannot express a chord*. Hitting that deliberately is what motivates Project 3.

**Test-set discipline, starts now and outlives every model:** split real photos into
`dev-real` (~50 pages, debug freely) and `test-real` (~50 pages, opened once per project).
You are a gradient descent process; looking at the same 50 photos daily overfits them through
your own choices. Annotate `test-real` before training and never add to it from observed failures.

---

<!-- BEGIN MUTABLE STATUS — the only section Claude may edit without asking. -->

## Status

**Current project:** 2 — CRNN + CTC on single staves.

**What exists:** `symbolic/` (transpose, spelling, instruments, render) + CLI. `datagen/`:
corpus loader (voice→single-staff, piano→grand-staff units), `Engraver` protocol with all
three engravers wired (Verovio/rsvg/white-bg, MuseScore 4, LilyPond), seeded augment
(photometric + geometric, mask-based crop safety), MusicXML labels, `write_dataset` +
manifest, `generate()`. `scripts/build_corpus.py` builds a diversity-weighted corpus from
the music21 core with `CORPUS.json` provenance (version, recipe, per-file sha256).
ADRs 0005–0009.

**What survived from Project 0:** the `Renderer` protocol (now with raster sibling `Engraver`,
sharing Verovio init); `spelling.normalize_spelling()` (also normalizes generation labels);
every verified `music21` trap in `test_traps.py`.

**Carried debt from Project 1 — two OPEN bugs, `augment.py`/`generate.py` unpatched.** Both
surfaced only against a *diverse* corpus; the short uniform Bach chorales never triggered them:
(1) multi-page works exceed OpenCV's `SHRT_MAX`, crashing `warpAffine`/`remap` in
`_apply_geometric` — needs a design call, skip vs. downscale; (2) `canonical_musicxml()` sits
*outside* the `try/except EngraverError` in `generate_with_meta`, so one unexportable score
(`MusicXMLExportException`) aborts a whole run instead of skipping the unit.
**Consequence: no end-to-end mixed dataset exists, so the three-engraver thesis has passing
tests but no generated set demonstrating it.** Project 2 stages 1–2 (PrIMuS, Camera-PrIMuS)
are unblocked and need no generator; **stage 3 is blocked until both bugs are fixed.**
Also deferred: per-unit multi-engraver sampling.

**Stage A (seam 4) is DONE.** `src/omrt/eval/` — `symbols.py` (PrIMuS-mirroring grammar),
`editdistance.py`, `metrics.py`/`evaluate()` — 33 tests, full suite 153 green, mypy strict
clean. Design + measured results: `docs/superpowers/specs/2026-07-24-project2-seam4-design.md`.

**Seam 4 bridge: the MEI route is LOSSY. Do not trust the earlier "verified" note.**
Calibration against the shipped `.semantic` (300 incipits, seed 0) reads **18.8% SER**, and
essentially all of it is music21's MEI parser, not our grammar: it ignores key-signature
accidentals (765 divergent ops — `Bb` read as `B`), drops `meter.sym` so every
`timeSignature-C` disappears (~101), and drops `<multiRest>` (92 + knock-on). The spike's
"300/300 note-count match" only ever showed the notes were *present*. A ~1,885-incipit sample
is at `data/primus_sample/package_aa/` (gitignored; re-fetch command in the spec).

**Settled (ADR 0010): bypass the MEI route, do not repair it.** PrIMuS `.semantic` is already
the token vocabulary; Stage B's `decode(tokens) -> MusicXML` reaches MusicXML with no MEI
involved, and the round-trip ceiling test measures that path directly. Consequence for Stage B:
seam 3 must decode the *full* `.semantic` grammar — there is no fallback route to MusicXML.
The 18.8% figure describes a route we no longer take; do not quote it as our metric's offset.

**Open question I'm stuck on:** none — next up is Stage B (seam 3), design gate not yet opened.

<!-- END MUTABLE STATUS -->

---

## Non-goals (Project 2)

Lifted from Project 1, deliberately, by human instruction: torch, models, and the training
dataloader are now in scope. Tokenized labels are in scope *inside* `decode/` only — seam 1
still emits MusicXML and nothing else.

- **Single-staff monophonic only.** No grand staff, no polyphony, no chords. CTC structurally
  cannot express them; attempting it is the Project 3 motivation, not a Project 2 task.
- **Do not reimplement from `calvozaragoza/tf-deep-omr`.** Build from the 2018 paper. The
  appendix grammars are what seam 3 needs. Reading the repo defeats the project.
- **Do not tune hyperparameters before the ceiling and canary tests pass.** A model that can't
  overfit 8 examples is broken, not undertrained.
- **Do not build a web UI, API, or mobile app.**
- No handwritten manuscript support, ever. Printed music only.
- No audio, no MIDI playback, no synthesis.
- Do not attempt to use an LLM to read notation from an image. It does not work and is not
  what this project is.

## Conventions

- Python 3.12+. `uv` for dependency management.
- Type hints everywhere. `mypy --strict` on `src/`.
- `pytest`. Prefer property-based tests (`hypothesis`) for anything music-theoretic —
  the invariants are far more informative than fixed examples.
- MusicXML strings, not file paths, across internal boundaries. Files only at the CLI edge.
- Commit early and often. Small commits with real messages.

## Maintaining this file

This file is loaded into every session. **Prefer to keep it lean** — every line costs context
in every session — but there is no hard limit, and it's fine to grow when new content earns its
place. It is a contract, not a changelog: the test for any addition is "would a fresh session be
wrong without this?", not "is there room?"

**Claude may edit, without asking:** the MUTABLE STATUS block only.

**Claude must ask before editing:** the four seams, the non-goals, the conventions, the
reality-check numbers. These are the load-bearing walls.

**Claude must never do this:** if a requested task conflicts with a non-goal, stop and say so.
Do not resolve the conflict by editing the non-goal. A rule that gets rewritten whenever it
becomes inconvenient was never a rule.

### What goes elsewhere

| Content | Home | Why not here |
|---|---|---|
| Benchmark numbers, run configs | `docs/benchmarks.csv` | grows without bound |
| "We chose X over Y because…" | `docs/decisions/NNNN-slug.md` (ADR, append-only) | history, not contract |
| What I learned, what died | `docs/retro/project-N.md` | narrative, not instruction |
| Anything a fresh session doesn't need | nowhere. delete it. | context is not free |

### End-of-project ritual

When a project completes, in this order:

1. Write `docs/retro/project-N.md`: what shipped, what got thrown away, what surprised you,
   what the numbers were.
2. **Prune this file.** Growth is allowed, but a project boundary is the moment to spend a
   few minutes deleting what's now dead: non-goals that no longer apply, conventions nobody
   follows, references you never opened. Add freely; just don't let stale content ride along.
3. Update the MUTABLE STATUS block: bump the project, record what survived.
4. Only then, add anything new — and only if a fresh session would be wrong without it.

### Triggers for editing the stable sections

Edit when, and only when:

- A seam signature changes. This should hurt. Everything downstream depends on it.
- A non-goal becomes a goal, deliberately, with a human saying so out loud.
- A convention changes (test framework, package manager, type-checking strictness).
- A published number in the reality-check table is superseded by newer literature.

Do **not** edit for: a new file, an experiment result, a fixed bug, a design discussion,
or a thing you learned. Those all have homes above.

---

## Key references

- Calvo-Zaragoza, Hajič, Pacha — *Understanding Optical Music Recognition*, ACM Computing
  Surveys 2020. The field's canonical survey.
- Calvo-Zaragoza & Rizo — *End-to-End Neural OMR of Monophonic Scores*, Applied Sciences 2018.
  The CRNN/CTC baseline we reimplement in Project 2.
- SMT++ — arXiv 2405.12105. Checkpoint: `PRAIG/smt-fp-grandstaff`.
- Legato — arXiv 2506.19065. Checkpoint: `guangyangmusic/legato-small` (MIT).
- Transcoda — arXiv 2605.10835. The data-centric argument.
- Dataset index: https://apacha.github.io/OMR-Datasets/
- PrIMuS: https://grfia.dlsi.ua.es/primus/ — 87,678 monodic single-staff scores, CC BY 4.0.