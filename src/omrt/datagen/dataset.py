"""Write N pairs to disk plus a manifest.

The manifest is a deliverable, not an afterthought: it is how we do coverage checks and how
we answer "did the model get worse, or the data?". Per sample it records the engraver used,
the exact augmentation parameters, and score-level histograms (key sig, time sig, staff
count, note density); it also carries corpus-wide aggregates over those axes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2

from omrt.datagen.config import GenConfig
from omrt.datagen.generate import SampleMeta, generate_with_meta

__all__ = ["write_dataset"]

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def write_dataset(n: int, config: GenConfig, out_dir: Path) -> Path:
    """Generate ``n`` pairs into ``out_dir`` and return the manifest path."""
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for i, (image, label, meta) in enumerate(generate_with_meta(n, config)):
        stem = f"{i:06d}_{_slug(meta.corpus_id)}"
        image_path = images_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.musicxml"
        cv2.imwrite(str(image_path), image)
        label_path.write_text(label, encoding="utf-8")
        meta = replace(
            meta,
            image_path=image_path.relative_to(out_dir).as_posix(),
            label_path=label_path.relative_to(out_dir).as_posix(),
        )
        samples.append(_meta_to_dict(meta))

    manifest = {
        "config": {
            "corpus_dir": str(config.corpus_dir),
            "seed": config.seed,
            "engravers": list(config.engravers),
            "dpi": config.dpi,
            "max_staves": config.max_staves,
            "photometric": config.photometric,
            "geometric": config.geometric,
            "config_hash": config.config_hash(),
        },
        "count": len(samples),
        "aggregate": _aggregate(samples),
        "samples": samples,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _slug(corpus_id: str) -> str:
    return _UNSAFE.sub("_", corpus_id)[:120]


def _meta_to_dict(meta: SampleMeta) -> dict[str, Any]:
    d = asdict(meta)
    return d


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage histograms across the whole set — the point of the manifest."""
    engravers: Counter[str] = Counter()
    key_sharps: Counter[Any] = Counter()
    time_sigs: Counter[Any] = Counter()
    staves: Counter[int] = Counter()
    units: Counter[str] = Counter()
    for s in samples:
        engravers[s["engraver"]] += 1
        hist = s["histograms"]
        key_sharps[hist.get("key_sharps")] += 1
        time_sigs[hist.get("time_sig")] += 1
        staves[hist.get("staves")] += 1
        units[hist.get("unit")] += 1
    return {
        "engraver": dict(engravers),
        "key_sharps": {str(k): v for k, v in sorted(key_sharps.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "time_sig": {str(k): v for k, v in time_sigs.items()},
        "staves": {str(k): v for k, v in sorted(staves.items())},
        "unit": dict(units),
    }
