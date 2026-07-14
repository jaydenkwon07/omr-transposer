"""Augmentation: seeded determinism and the crop-safety invariant.

Crop safety is the load-bearing one — a system dropped from the image while the label
still claims it is silent and poisonous. The invariant (ADR 0005) is that no ink leaves
the frame, checked by pushing the ink mask through the same geometric pipeline as the
image. These tests exercise it directly on synthetic pages so they are fast and hermetic.
"""
from __future__ import annotations

import numpy as np
import pytest

from omrt.datagen.augment import CropSafetyError, augment, ink_mask
from omrt.datagen.config import GenConfig


def _page_with_two_systems() -> np.ndarray:
    """A white page with two horizontal ink bars — two 'systems', near top and bottom."""
    img = np.full((400, 300), 255, dtype=np.uint8)
    img[40:70, 30:270] = 0     # first system
    img[330:360, 30:270] = 0   # last system
    return img


def _cfg(**kw: object) -> GenConfig:
    from pathlib import Path
    return GenConfig(corpus_dir=Path("."), **kw)  # corpus_dir unused by augment


def test_augment_is_deterministic_for_a_seed() -> None:
    img = _page_with_two_systems()
    cfg = _cfg()
    a, pa = augment(img, np.random.default_rng(123), cfg)
    b, pb = augment(img, np.random.default_rng(123), cfg)
    assert np.array_equal(a, b)
    assert pa == pb


def test_augment_differs_across_seeds() -> None:
    img = _page_with_two_systems()
    cfg = _cfg()
    a, _ = augment(img, np.random.default_rng(1), cfg)
    b, _ = augment(img, np.random.default_rng(2), cfg)
    # Different geometry/photometry -> different shape or pixels (overwhelmingly likely).
    assert a.shape != b.shape or not np.array_equal(a, b)


@pytest.mark.parametrize("seed", range(40))
def test_crop_never_drops_a_system(seed: int) -> None:
    """Geometry only (photometric off so ink stays dark): every ink pixel of the label
    must survive. A dropped system would show up as lost ink."""
    img = _page_with_two_systems()
    before = int((ink_mask(img) > 0).sum())
    out, _ = augment(img, np.random.default_rng(seed), _cfg(photometric=False))
    after = int((out < 128).sum())
    # Nearest-neighbour warping perturbs the count slightly; a dropped system would
    # remove ~half. Require the vast majority to survive.
    assert after >= 0.9 * before, f"seed {seed}: lost ink {before} -> {after}"


def test_crop_guard_catches_out_of_frame_ink() -> None:
    """If ink reaches the very edge of the page, the pad+warp guard must fire rather than
    silently clip a system."""
    img = np.full((200, 200), 255, dtype=np.uint8)
    img[:, :] = 0  # entirely ink: warping cannot avoid pushing ink past the padded edge
    with pytest.raises(CropSafetyError):
        # zero pad removes the safety margin, forcing the guardrail to trigger
        augment(img, np.random.default_rng(0), _cfg(photometric=False, warp_pad=0.0))
