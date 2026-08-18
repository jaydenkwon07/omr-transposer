import numpy as np
import pytest

pytest.importorskip("torch")

from omrt.models.dataset import preprocess


def test_preprocess_fixes_height_and_preserves_aspect():
    img = np.full((64, 200), 255, dtype=np.uint8)  # paper
    img[10:20, :] = 0  # an ink band
    t = preprocess(img)
    assert t.shape[0] == 1
    assert t.shape[1] == 128
    # 200 * (128/64) = 400
    assert t.shape[2] == 400
    assert t.dtype.is_floating_point
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
