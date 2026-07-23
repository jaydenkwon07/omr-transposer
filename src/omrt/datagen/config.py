"""Generation configuration and its reproducibility hash.

``config_hash`` is the ``config_hash`` half of the ``(corpus_id, seed, config_hash)``
triple that every pair must be reproducible from. It deliberately excludes ``corpus_dir``
(an absolute path is machine-specific) and ``seed`` (its own axis of the triple), so the
same logical configuration hashes identically on any machine.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# All three are wired end-to-end (ADR 0009). Each depends on external binaries, so any
# whose backend is missing on this machine reports available() False and build_engravers
# drops it silently — a Verovio-only box still generates, just without the diversity.
DEFAULT_ENGRAVERS: tuple[str, ...] = ("verovio", "lilypond", "musescore")


@dataclass(frozen=True)
class GenConfig:
    """Everything that determines what ``generate`` produces, besides the seed."""

    corpus_dir: Path
    seed: int = 0
    engravers: tuple[str, ...] = DEFAULT_ENGRAVERS
    dpi: int = 150
    max_staves: int = 2  # single-staff (1) + piano grand staff (2) only, per CLAUDE.md
    photometric: bool = True
    geometric: bool = True
    # Extra margin (fraction of page) padded before geometric warps so rotation and
    # perspective have room to move content without clipping at the canvas edge.
    warp_pad: float = 0.15

    def config_hash(self) -> str:
        payload = {
            "engravers": sorted(self.engravers),
            "dpi": self.dpi,
            "max_staves": self.max_staves,
            "photometric": self.photometric,
            "geometric": self.geometric,
            "warp_pad": round(self.warp_pad, 6),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
