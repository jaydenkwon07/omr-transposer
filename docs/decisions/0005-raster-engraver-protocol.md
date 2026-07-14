# 0005 — A separate raster Engraver protocol (and rsvg, not cairosvg)

Status: accepted (Project 1, 2026-07-13)

## Context

Project 1 needs to rasterize MusicXML for training images. Project 0 already has a
`Renderer` protocol (`to_pdf(musicxml) -> bytes`, SVG → PDF via `rsvg-convert`, ADR 0001).
The tempting move is to reuse it. But the two have different consumers, lifecycles, and
return types: `Renderer` serves the shipped product and returns a document; the training
engraver serves the data pipeline, whose very next step is augmentation in pixel space.

## Decision

A new `Engraver` protocol in `datagen/engravers/`, distinct from `Renderer`:

- **Returns an in-memory grayscale `uint8` ndarray** (255=paper, 0=ink), not bytes and not
  a PIL image. Sheet music is single-channel and the model consumes grayscale; committing
  to one channel representation from the engraver outward also sidesteps the RGB-vs-BGR
  footgun between PIL and cv2. Returning PNG bytes would just force `augment` to decode
  them again immediately.
- **`name: str` lives on the protocol.** The manifest must record which engraver made each
  sample; putting `name` on the type means an engraver cannot exist without identifying
  itself — the requirement is enforced by the compiler, not by remembering to log.
- **Project 0's `Renderer` is left untouched.** It is a tested, shipped seam.
- Verovio toolkit init is shared with `VerovioRenderer` via one private helper
  (`render.render_verovio_svgs`) — shared implementation, not a shared interface.

**Tool deviation:** CLAUDE.md's Project 1 note names cairosvg for SVG → PNG. We use
`rsvg-convert -f png -b white` instead. cairosvg needs a loadable native `libcairo` that is
absent on this machine — the same loader-path problem that made Project 0 pick rsvg-convert
in ADR 0001. The *verified-critical* constraint is the **white background** (transparent
SVG rasterizes to solid black in grayscale), and rsvg preserves it with `-b white`. The
tool differs; the constraint does not.

## Consequences

- Two clean protocols, two clean consumer sets. A future engraver implements `Engraver`
  and is automatically manifest-compatible.
- One less native dependency to install for datagen (no libcairo); rsvg-convert is already
  required by Project 0.
- If a machine ever needs cairosvg specifically, that is a new backend, not a change here.
