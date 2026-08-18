import pytest

pytest.importorskip("torch")

import torch

from omrt.models.crnn import CRNN


def test_forward_shape_matches_frame_math():
    model = CRNN(vocab_size=50)
    x = torch.rand(2, 1, 128, 160)  # W=160 -> T=40
    out = model(x)
    assert out.shape == (40, 2, 50)


def test_output_is_log_probabilities():
    model = CRNN(vocab_size=10)
    out = model(torch.rand(1, 1, 128, 80))
    probs = out.exp().sum(dim=2)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)
