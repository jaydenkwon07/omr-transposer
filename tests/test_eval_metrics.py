"""Seam 4: symbol error rate over MusicXML.

The metric is built and tested before any decoder or model exists — a buggy metric
reports improvement that isn't there.
"""
from __future__ import annotations

from omrt.eval import levenshtein


def test_levenshtein_identity_is_zero() -> None:
    ops = levenshtein(["a", "b", "c"], ["a", "b", "c"])
    assert ops.distance == 0
    assert (ops.substitutions, ops.insertions, ops.deletions) == (0, 0, 0)


def test_levenshtein_counts_substitutions() -> None:
    ops = levenshtein(["a", "b", "c"], ["a", "X", "c"])
    assert ops.distance == 1
    assert (ops.substitutions, ops.insertions, ops.deletions) == (1, 0, 0)


def test_levenshtein_counts_insertions() -> None:
    # b (the reference) has one symbol a does not: reaching b requires an insertion.
    ops = levenshtein(["a", "c"], ["a", "b", "c"])
    assert ops.distance == 1
    assert (ops.substitutions, ops.insertions, ops.deletions) == (0, 1, 0)


def test_levenshtein_counts_deletions() -> None:
    ops = levenshtein(["a", "b", "c"], ["a", "c"])
    assert ops.distance == 1
    assert (ops.substitutions, ops.insertions, ops.deletions) == (0, 0, 1)


def test_levenshtein_empty_against_nonempty_is_all_insertions() -> None:
    ops = levenshtein([], ["a", "b"])
    assert ops.distance == 2
    assert (ops.substitutions, ops.insertions, ops.deletions) == (0, 2, 0)


def test_levenshtein_op_counts_sum_to_distance() -> None:
    ops = levenshtein(["a", "b", "c", "d"], ["a", "X", "c", "d", "e", "f"])
    assert ops.substitutions + ops.insertions + ops.deletions == ops.distance
