"""Calibrate our symbol stream against PrIMuS's shipped ``.semantic`` ground truth.

For each incipit we hold both the published token sequence and our own
``to_symbols()`` output over the same music, so the diff between them is the standing
offset between this metric and the one the 2018 paper reports — measured before any
model exists.

The route from PrIMuS to MusicXML is ``.mei`` -> music21 -> MusicXML, so what this
measures is the *pair*: our grammar plus that bridge. Divergence attributable to the
bridge is broken out per token class below, because the two have very different fixes.

Usage::

    uv run python scripts/calibrate_primus.py [--sample N] [--seed S] [--root DIR]
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from music21 import converter
from music21.musicxml.m21ToXml import GeneralObjectExporter

from omrt.eval import levenshtein, to_symbols

DEFAULT_ROOT = Path("data/primus_sample/package_aa")


def _token_class(token: str) -> str:
    return token.split("-", 1)[0] if "-" in token else token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    incipits = sorted(p for p in args.root.iterdir() if p.is_dir())
    if not incipits:
        print(f"no incipits under {args.root}", file=sys.stderr)
        return 1
    # Sample randomly, not head -N: the corpus is ordered by catalogue number, and the
    # first N share sources, keys and lengths (see memory: project-1-open-bugs).
    if args.sample and args.sample < len(incipits):
        incipits = random.Random(args.seed).sample(incipits, args.sample)

    exact = failed = 0
    total_edits = total_ref = 0
    per_incipit_ser: list[float] = []
    class_divergence: Counter[str] = Counter()

    for d in incipits:
        semantic_file = d / f"{d.name}.semantic"
        mei_file = d / f"{d.name}.mei"
        if not semantic_file.exists() or not mei_file.exists():
            continue
        truth = semantic_file.read_text().strip().split("\t")
        try:
            score = converter.parse(str(mei_file))
            musicxml = GeneralObjectExporter().parse(score).decode("utf-8")
        except Exception:  # noqa: BLE001 — a bridge failure is a calibration data point
            failed += 1
            continue

        ours = to_symbols(musicxml)
        ops = levenshtein(ours, truth)
        total_edits += ops.distance
        total_ref += len(truth)
        per_incipit_ser.append(ops.distance / len(truth) if truth else 0.0)
        if ops.distance == 0:
            exact += 1

        # Attribute divergence to a token class by comparing per-class multisets. This
        # is coarser than the alignment but says *which* symbols disagree, which is the
        # actionable part.
        ours_by_class: Counter[str] = Counter(_token_class(t) for t in ours)
        truth_by_class: Counter[str] = Counter(_token_class(t) for t in truth)
        for name in set(ours_by_class) | set(truth_by_class):
            class_divergence[name] += abs(ours_by_class[name] - truth_by_class[name])
        # Same class, different spelling: count the leftover disagreement as "spelling".
        common = (Counter(ours) & Counter(truth)).total()
        class_divergence["_exact_token_overlap"] += common
        class_divergence["_truth_tokens"] += len(truth)

    scored = len(per_incipit_ser)
    print(f"incipits scored          {scored}")
    print(f"bridge failures          {failed}")
    print(f"exact match              {exact} ({exact / scored:.1%})" if scored else "")
    print(f"corpus SER               {total_edits / total_ref:.4f}" if total_ref else "")
    print(
        f"mean per-incipit SER     {sum(per_incipit_ser) / scored:.4f}" if scored else ""
    )
    overlap = class_divergence.pop("_exact_token_overlap", 0)
    truth_tokens = class_divergence.pop("_truth_tokens", 0)
    if truth_tokens:
        print(f"token-level overlap      {overlap / truth_tokens:.1%}")
    print("\ncount divergence by token class (|ours - truth|, summed):")
    for name, count in class_divergence.most_common():
        print(f"  {name:16s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
