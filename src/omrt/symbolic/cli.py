"""Command-line entry point.

The only module in ``symbolic`` that touches files. Everything inside the package
boundary passes MusicXML *strings*.

    omrt transpose IN.musicxml --to-key "F# major"   --out OUT.pdf
    omrt transpose IN.musicxml --by-interval m-3      --out OUT.pdf
    omrt transpose IN.musicxml --for-instrument bb-trumpet --out OUT.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Sequence

from omrt.symbolic.render import RenderError, Renderer, VerovioRenderer
from omrt.symbolic.transpose import (
    transpose_by_interval,
    transpose_for_instrument,
    transpose_to_key,
)

# Flags whose value may legitimately begin with '-' (a descending interval like -m3).
# argparse would treat that value as an option, so we coalesce "FLAG value" into
# "FLAG=value" before parsing. This is the fix for the leading-hyphen problem.
_VALUE_FLAGS = frozenset({"--to-key", "--by-interval", "--for-instrument", "--out", "-o"})


def _coalesce_negative_values(argv: Sequence[str]) -> list[str]:
    out: list[str] = []
    i = 0
    argv = list(argv)
    while i < len(argv):
        tok = argv[i]
        if (
            tok in _VALUE_FLAGS
            and i + 1 < len(argv)
            and argv[i + 1].startswith("-")
            and argv[i + 1] != "--"
        ):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
        else:
            out.append(tok)
            i += 1
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omrt", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("transpose", help="transpose a MusicXML score and render a PDF")
    tr.add_argument("input", type=Path, help="input MusicXML file")
    tr.add_argument("--out", "-o", type=Path, required=True, help="output PDF path")

    mode = tr.add_mutually_exclusive_group(required=True)
    mode.add_argument("--to-key", metavar="KEY", help='target key, e.g. "F# major"')
    mode.add_argument(
        "--by-interval",
        metavar="INTERVAL",
        help="interval, e.g. m3, P5; descending as m-3 or -m3",
    )
    mode.add_argument(
        "--for-instrument",
        metavar="NAME",
        help="transposing instrument, e.g. bb-trumpet",
    )
    return parser


def _transpose(args: argparse.Namespace, musicxml: str) -> str:
    if args.to_key is not None:
        return transpose_to_key(musicxml, args.to_key)
    if args.by_interval is not None:
        return transpose_by_interval(musicxml, args.by_interval)
    return transpose_for_instrument(musicxml, args.for_instrument)


def main(argv: Sequence[str] | None = None, renderer: Renderer | None = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    parser = _build_parser()
    args = parser.parse_args(_coalesce_negative_values(raw))

    renderer = renderer or VerovioRenderer()

    try:
        musicxml = args.input.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"omrt: cannot read {args.input}: {exc}", file=sys.stderr)
        return 1

    try:
        transposed = _transpose(args, musicxml)
        pdf = renderer.to_pdf(transposed)
    except (ValueError, RenderError) as exc:
        print(f"omrt: {exc}", file=sys.stderr)
        return 1

    try:
        args.out.write_bytes(pdf)
    except OSError as exc:
        print(f"omrt: cannot write {args.out}: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
