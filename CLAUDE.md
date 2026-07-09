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

<!-- BEGIN MUTABLE STATUS — the only section Claude may edit without asking. -->

## Status

**Current project:** 0 — symbolic layer.

**What exists:** nothing yet.

**What survived from previous projects:** n/a.

**Open question I'm stuck on:** n/a.

<!-- END MUTABLE STATUS -->

---

## Non-goals (right now)

- **Do not scaffold `models/`, `datagen/`, or `eval/`.** Do not install torch. Do not create
  placeholder files for future projects. Empty directories with `.gitkeep` are acceptable;
  stub classes are not.
- **Do not build a web UI, API, or mobile app.** Project 0 is a CLI.
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

This file is loaded into every session. It must stay **under 200 lines**. It is a contract,
not a changelog.

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
2. **Prune this file.** Default gravity is accretion, so make the ritual explicitly about
   deletion. Remove non-goals that no longer apply. Remove conventions nobody follows.
   Remove references you never opened. If the file grew, you did it wrong.
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