"""Levenshtein distance with per-operation counts.

Pure Python, no dependencies. Sequences here are symbol lists of incipit length
(tens to a few hundred), so the full DP matrix is cheap and a backtrace is simpler
to trust than op counters threaded through the recurrence.

Direction convention: ``levenshtein(a, b)`` counts the edits that turn ``a`` into
``b``. An *insertion* adds a symbol ``b`` has and ``a`` lacks; a *deletion* drops a
symbol ``a`` has and ``b`` lacks. ``distance`` is symmetric in ``(a, b)``; the
individual op counts are not — insertions and deletions swap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EditOps:
    """One alignment's edit tally. ``substitutions + insertions + deletions == distance``."""

    distance: int
    substitutions: int
    insertions: int
    deletions: int


def levenshtein(a: Sequence[str], b: Sequence[str]) -> EditOps:
    """Edit distance from ``a`` to ``b``, with the operations broken out."""
    n, m = len(a), len(b)

    # d[i][j] = distance from a[:i] to b[:j].
    d: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        row, prev = d[i], d[i - 1]
        a_i = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if a_i == b[j - 1] else 1
            row[j] = min(prev[j - 1] + cost, prev[j] + 1, row[j - 1] + 1)

    # Backtrace, preferring the diagonal so that equal-cost alignments are scored as
    # substitutions rather than an insert/delete pair. Distance is unaffected either way;
    # this only makes the reported breakdown the intuitive one.
    subs = ins = dels = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                subs += cost
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels += 1
            i -= 1
            continue
        ins += 1
        j -= 1

    return EditOps(distance=d[n][m], substitutions=subs, insertions=ins, deletions=dels)
