from __future__ import annotations

import torch
from torch import Tensor

_BLANK = 0


def ctc_greedy_decode(log_probs: Tensor, input_lengths: Tensor) -> list[list[int]]:
    """Greedy CTC decode (paper §4): per frame argmax, collapse consecutive duplicates,
    drop the blank. `log_probs` is [T, N, C]; only the first input_lengths[i] frames of
    sample i are considered."""
    best = log_probs.argmax(dim=2)  # [T, N]
    results: list[list[int]] = []
    for i in range(best.shape[1]):
        length = int(input_lengths[i])
        prev = -1
        seq: list[int] = []
        for f in range(length):
            c = int(best[f, i])
            if c != prev and c != _BLANK:
                seq.append(c)
            prev = c
        results.append(seq)
    return results
