from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import Tensor

from omrt.datagen.types import Image

_TARGET_HEIGHT = 128


def preprocess(image: Image) -> Tensor:
    """Grayscale staff -> [1, 128, W] float tensor, aspect ratio preserved, values in [0,1]
    (paper≈1.0, ink≈0.0). Height is fixed at 128 (paper Table 2)."""
    h, w = image.shape[:2]
    new_w = max(1, round(w * (_TARGET_HEIGHT / h)))
    resized = cv2.resize(image, (new_w, _TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)
