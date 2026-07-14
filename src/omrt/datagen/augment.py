"""Composable, seeded augmentation.

Two groups: **photometric** (lighting gradient, shadow, blur, JPEG artifact, moiré) which
leave ink where it is, and **geometric** (perspective, page curl, rotation, crop) which
move it. Every transform draws from a single ``np.random.Generator`` — never global
``np.random`` — so a run is reproducible from one seed.

Crop safety (ADR 0007): a system dropped from the image while the label still claims it is
this project's silent killer, and it is not a property of the crop alone — a rotation or
perspective warp can push content out of frame on its own. So we build an *ink mask* of the
pre-augmentation page and push it through the **exact same composed geometric pipeline** as
the image; a dropped system is, by definition, mask pixels that left the frame. We assert
none do. v1 consequence, accepted deliberately: no ink ever leaves the frame, so the model
never sees partial-page photos (a cut-off last system, a finger over a measure). That is a
real gap, not a bug — the realistic version crops *into* content and crops the label to
match. See ADR 0007.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from omrt.datagen.config import GenConfig
from omrt.datagen.types import Image

__all__ = ["augment", "AugmentParams", "CropSafetyError", "ink_mask"]

_INK_BELOW = 250  # a pixel darker than this counts as ink
_PAPER = 255


class CropSafetyError(RuntimeError):
    """The geometric pipeline pushed ink outside the final frame — a system would be
    dropped while the label still claims it."""


@dataclass(frozen=True)
class AugmentParams:
    """Every sampled value, recorded verbatim in the manifest."""

    photometric: dict[str, Any] = field(default_factory=dict)
    geometric: dict[str, Any] = field(default_factory=dict)


def ink_mask(img: Image) -> Image:
    """255 where there is ink, 0 on paper."""
    return np.where(img < _INK_BELOW, np.uint8(255), np.uint8(0)).astype(np.uint8)


def augment(img: Image, rng: np.random.Generator, config: GenConfig) -> tuple[Image, AugmentParams]:
    """Apply the enabled augmentation groups and return the page plus its parameters."""
    geo: dict[str, Any] = {}
    photo: dict[str, Any] = {}
    out = img
    if config.geometric:
        out, geo = _apply_geometric(out, rng, config.warp_pad)
    if config.photometric:
        out, photo = _apply_photometric(out, rng)
    return out, AugmentParams(photometric=photo, geometric=geo)


# --- geometric ---------------------------------------------------------------

def _apply_geometric(img: Image, rng: np.random.Generator, pad_frac: float) -> tuple[Image, dict[str, Any]]:
    mask = ink_mask(img)
    pad_v = int(round(img.shape[0] * pad_frac))
    pad_h = int(round(img.shape[1] * pad_frac))
    img_p = cv2.copyMakeBorder(img, pad_v, pad_v, pad_h, pad_h, cv2.BORDER_CONSTANT, value=_PAPER)
    mask_p = cv2.copyMakeBorder(mask, pad_v, pad_v, pad_h, pad_h, cv2.BORDER_CONSTANT, value=0)

    # Sample every parameter up front so the image and mask see an identical pipeline.
    angle = float(rng.uniform(-3.0, 3.0))
    corner = rng.uniform(-0.03, 0.03, size=(4, 2)).astype(float)
    curl_amp = float(rng.uniform(0.0, 0.02) * img_p.shape[0])

    img_g = _curl(_perspective(_rotate(img_p, angle, _PAPER, cv2.INTER_LINEAR),
                               corner, _PAPER, cv2.INTER_LINEAR),
                  curl_amp, _PAPER, cv2.INTER_LINEAR)
    mask_g = _curl(_perspective(_rotate(mask_p, angle, 0, cv2.INTER_NEAREST),
                                corner, 0, cv2.INTER_NEAREST),
                   curl_amp, 0, cv2.INTER_NEAREST)

    # Guardrail 1: the warp itself must not have clipped ink at the padded edge.
    if mask_g[0, :].any() or mask_g[-1, :].any() or mask_g[:, 0].any() or mask_g[:, -1].any():
        raise CropSafetyError("geometric warp pushed ink to the canvas edge")

    top, left, bottom, right = _ink_bbox(mask_g)
    h, w = mask_g.shape
    # Crop must contain the whole ink bbox; randomize only within the surrounding margin.
    ct = int(rng.integers(0, top + 1))
    cl = int(rng.integers(0, left + 1))
    cb = int(rng.integers(bottom, h + 1))
    cr = int(rng.integers(right, w + 1))

    # Guardrail 2: no ink may fall outside the chosen crop.
    if mask_g[:ct, :].any() or mask_g[cb:, :].any() or mask_g[:, :cl].any() or mask_g[:, cr:].any():
        raise CropSafetyError("crop would drop ink present in the label")

    out = img_g[ct:cb, cl:cr].astype(np.uint8)
    params = {
        "rotation_deg": round(angle, 4),
        "perspective_corner": [[round(v, 4) for v in row] for row in corner.tolist()],
        "curl_amplitude_px": round(curl_amp, 2),
        "crop": [ct, cl, cb, cr],
    }
    return out, params


def _rotate(img: Image, angle: float, border: int, flags: int) -> Image:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return np.asarray(out, dtype=np.uint8)


def _perspective(img: Image, corner: np.ndarray, border: int, flags: int) -> Image:
    h, w = img.shape
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + corner.astype(np.float32) * np.array([w, h], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, m, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return np.asarray(out, dtype=np.uint8)


def _curl(img: Image, amp: float, border: int, flags: int) -> Image:
    if amp <= 0:
        return img
    h, w = img.shape
    xs = np.arange(w, dtype=np.float32)
    map_x = np.tile(xs, (h, 1)).astype(np.float32)
    row_shift = (amp * np.sin(2.0 * np.pi * xs / w)).astype(np.float32)
    map_y = (np.arange(h, dtype=np.float32)[:, None] + row_shift[None, :]).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, interpolation=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return np.asarray(out, dtype=np.uint8)


def _ink_bbox(mask: Image) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise CropSafetyError("unit rendered with no ink; cannot bound systems")
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


# --- photometric -------------------------------------------------------------

def _apply_photometric(img: Image, rng: np.random.Generator) -> tuple[Image, dict[str, Any]]:
    f = img.astype(np.float32)
    lo = float(rng.uniform(0.55, 0.85))
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    f = _lighting_gradient(f, lo, ang)
    shadow = float(rng.uniform(0.3, 0.6))
    scorner = int(rng.integers(0, 4))
    f = _shadow(f, shadow, scorner)
    sigma = float(rng.uniform(0.0, 1.2))
    if sigma > 0.1:
        f = cv2.GaussianBlur(f, (0, 0), sigma)
    amp = float(rng.uniform(0.0, 6.0))
    period = float(rng.uniform(3.0, 9.0))
    mang = float(rng.uniform(0.0, np.pi))
    f = _moire(f, amp, period, mang)
    out = np.clip(f, 0, 255).astype(np.uint8)
    quality = int(rng.integers(30, 91))
    out = _jpeg_artifact(out, quality)
    return out, {
        "light_low": round(lo, 3),
        "light_angle": round(ang, 3),
        "shadow_strength": round(shadow, 3),
        "shadow_corner": scorner,
        "blur_sigma": round(sigma, 3),
        "moire_amp": round(amp, 3),
        "moire_period": round(period, 3),
        "moire_angle": round(mang, 3),
        "jpeg_quality": quality,
    }


def _ramp(h: int, w: int, angle: float) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = np.cos(angle) * xs / max(w - 1, 1) + np.sin(angle) * ys / max(h - 1, 1)
    return np.asarray((proj - proj.min()) / (np.ptp(proj) + 1e-6), dtype=np.float32)


def _lighting_gradient(f: np.ndarray, lo: float, angle: float) -> np.ndarray:
    h, w = f.shape
    scale = lo + (1.0 - lo) * _ramp(h, w, angle)
    return np.asarray(f * scale, dtype=np.float32)


def _shadow(f: np.ndarray, strength: float, corner: int) -> np.ndarray:
    h, w = f.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = 0.0 if corner in (0, 3) else float(w)
    cy = 0.0 if corner in (0, 1) else float(h)
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / np.sqrt(w ** 2 + h ** 2)
    darken = 1.0 - strength * np.clip(1.0 - dist / 0.6, 0.0, 1.0)
    return np.asarray(f * darken, dtype=np.float32)


def _moire(f: np.ndarray, amp: float, period: float, angle: float) -> np.ndarray:
    if amp <= 0:
        return f
    h, w = f.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = np.cos(angle) * xs + np.sin(angle) * ys
    return np.asarray(f + amp * np.sin(2.0 * np.pi * proj / period), dtype=np.float32)


def _jpeg_artifact(img: Image, quality: int) -> Image:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return np.asarray(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
