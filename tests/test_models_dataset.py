import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from omrt.models.dataset import (
    _TRAIL_PAD,
    PrimusDataset,
    collate_fn,
    iter_samples,
    list_incipit_ids,
    preprocess,
)
from omrt.models.vocab import Vocabulary

_SAMPLE = "data/primus_sample/package_aa"


def test_preprocess_fixes_height_and_preserves_aspect():
    img = np.full((64, 200), 255, dtype=np.uint8)  # paper
    img[10:20, :] = 0  # an ink band
    t = preprocess(img)
    assert t.shape[0] == 1
    assert t.shape[1] == 128
    # 200 * (128/64) = 400 content columns, plus the right-side white commit margin.
    assert t.shape[2] == 400 + _TRAIL_PAD
    # the appended margin is white (paper=1.0) so CTC greedy can commit the final label.
    assert bool((t[0, :, 400:] == 1.0).all())
    assert t.dtype.is_floating_point
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


@pytest.mark.skipif(not list_incipit_ids(_SAMPLE), reason="PrIMuS sample absent")
def test_iter_samples_reads_png_and_tokens():
    ids = list_incipit_ids(_SAMPLE)[:3]
    got = list(iter_samples(_SAMPLE, ids))
    assert len(got) == 3
    for image, tokens in got:
        assert image.ndim == 2 and image.dtype == np.uint8
        assert tokens and all(isinstance(t, str) for t in tokens)


@pytest.mark.skipif(not list_incipit_ids(_SAMPLE), reason="PrIMuS sample absent")
def test_dataset_item_satisfies_width_constraint():
    ids = list_incipit_ids(_SAMPLE)[:10]
    vocab = Vocabulary.build(t for _, t in iter_samples(_SAMPLE, ids))
    ds = PrimusDataset(_SAMPLE, ids, vocab)
    for i in range(len(ds)):
        image, target = ds[i]
        assert image.shape[1] == 128
        assert image.shape[2] // 4 >= target.shape[0]  # W/4 >= L
        assert target.dtype == torch.int64


def test_collate_pads_width_and_reports_lengths():
    a = (torch.ones(1, 128, 40), torch.tensor([1, 2, 3], dtype=torch.int64))
    b = (torch.ones(1, 128, 80), torch.tensor([4, 5], dtype=torch.int64))
    images, targets, input_lengths, target_lengths = collate_fn([a, b])
    assert images.shape == (2, 1, 128, 80)  # padded to max width
    assert list(target_lengths) == [3, 2]
    assert list(input_lengths) == [10, 20]  # W_i // 4
    assert list(targets) == [1, 2, 3, 4, 5]  # concatenated
