from __future__ import annotations

import cv2
import pytest

pytest.importorskip("torch")

from omrt.models.dataset import iter_samples, list_incipit_ids, preprocess

_SAMPLE = "data/primus_sample/package_aa"


def _png_path(incipit_id: str) -> str:
    return f"{_SAMPLE}/{incipit_id}/{incipit_id}.png"


@pytest.mark.skipif(not list_incipit_ids(_SAMPLE), reason="PrIMuS sample absent")
def test_natural_width_supports_label_length_for_whole_sample() -> None:
    all_ids = list_incipit_ids(_SAMPLE)

    # Pre-filter to PNGs that actually decode. The truncated sample has exactly one
    # known-bad file (libpng CRC error, an artifact of the ranged fetch); the full
    # 273MB corpus won't have it. read_gray_over_white raises on it, so we exclude it
    # here rather than letting iter_samples crash the whole guard.
    readable_ids = [i for i in all_ids if cv2.imread(_png_path(i), cv2.IMREAD_UNCHANGED) is not None]
    corrupt_count = len(all_ids) - len(readable_ids)

    total = len(all_ids)
    assert corrupt_count / total < 0.01, (
        f"{corrupt_count}/{total} PNGs failed to decode; expected at most the one known-bad "
        f"fetch artifact. This ratio catches a mass-corruption regression in the sample."
    )

    violations = []
    for incipit_id, (image, tokens) in zip(readable_ids, iter_samples(_SAMPLE, readable_ids)):
        frames = preprocess(image).shape[2] // 4
        if frames < len(tokens):
            violations.append((incipit_id, frames, len(tokens)))

    readable = len(readable_ids)
    print(
        f"total={total} corrupt={corrupt_count} readable={readable} "
        f"violations={len(violations)} ratio={len(violations) / readable:.5f}"
    )

    # A handful of pathologically dense incipits may need padding; a systemic failure
    # (more than 1%) means the pooling is too aggressive for this vocabulary.
    assert len(violations) / readable < 0.01, (
        f"{len(violations)}/{readable} incipits have natural W/4 < label length; "
        f"first few: {violations[:5]}"
    )
