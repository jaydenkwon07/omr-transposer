from __future__ import annotations

import glob
import os

import pytest

from omrt.decode.semantic import decode, parse_semantic
from omrt.eval.editdistance import levenshtein
from omrt.eval.symbols import to_symbols

_CORPUS = "data/primus_sample/package_aa"


def _incipits() -> list[str]:
    return sorted(glob.glob(os.path.join(_CORPUS, "*", "*.semantic")))


def _strip_trailing_barline(seq: list[str]) -> list[str]:
    # to_symbols always emits a barline after the final measure; ~43% of PrIMuS
    # .semantic incipits omit it. That single-token convention difference carries no
    # musical information and cancels in real model eval (truth is decoded through the
    # same path), so reconcile it before the NORMALIZED metric.
    return seq[:-1] if seq and seq[-1] == "barline" else seq


@pytest.mark.skipif(not _incipits(), reason="PrIMuS sample absent; see seam-4 spec for re-fetch")
def test_round_trip_ceiling_over_corpus():
    files = _incipits()
    raw_ops = raw_len = 0
    norm_ops = norm_len = 0
    raw_mismatch = 0
    norm_mismatches: list[tuple[str, list[str], list[str]]] = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            tokens = parse_semantic(fh.read())
        if not tokens:
            continue
        got = to_symbols(decode(tokens))
        raw_ops += levenshtein(tokens, got).distance
        raw_len += len(tokens)
        if got != tokens:
            raw_mismatch += 1
        t_norm = _strip_trailing_barline(tokens)
        g_norm = _strip_trailing_barline(got)
        norm_ops += levenshtein(t_norm, g_norm).distance
        norm_len += len(t_norm)
        if g_norm != t_norm:
            norm_mismatches.append((path, t_norm, g_norm))
    raw_ser = raw_ops / raw_len if raw_len else 0.0
    norm_ser = norm_ops / norm_len if norm_len else 0.0
    norm_exact = len(files) - len(norm_mismatches)
    print(f"\nceiling RAW:        SER={raw_ser:.4%}  mismatch={raw_mismatch}/{len(files)}")
    print(f"ceiling NORMALIZED: SER={norm_ser:.4%}  exact={norm_exact}/{len(files)}"
          f"  mismatch={len(norm_mismatches)}")
    # Surface the NORMALIZED mismatches (the interesting, non-artifact ones) to categorize.
    for path, want, got in norm_mismatches[:30]:
        print(f"  {os.path.basename(path)}:\n    want {want}\n    got  {got}")
    assert norm_ser < 0.005, (
        f"normalized ceiling SER {norm_ser:.4%} exceeds 0.5% budget; "
        f"categorize the {len(norm_mismatches)} normalized mismatches before relaxing"
    )
