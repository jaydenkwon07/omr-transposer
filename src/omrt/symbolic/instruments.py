"""Transposing-instrument registry.

Each factory returns a fresh music21 ``Instrument`` with its ``transposition`` set
**explicitly** — the interval from written to sounding pitch. We never rely on the
class default: those defaults vary across music21 versions (e.g. ``Trumpet()`` has
shipped as both a C and a B-flat instrument), so setting it here is the only
version-robust guarantee.
"""

from __future__ import annotations

from collections.abc import Callable

from music21 import instrument, interval

__all__ = ["INSTRUMENTS", "resolve_instrument"]


def _make(
    cls: type[instrument.Instrument], transposition: str
) -> instrument.Instrument:
    inst = cls()
    inst.transposition = interval.Interval(transposition)
    return inst


# name -> factory. Transposition is written->sounding, so a B-flat instrument
# (written C sounds a major second lower) is "M-2".
INSTRUMENTS: dict[str, Callable[[], instrument.Instrument]] = {
    "bb-trumpet": lambda: _make(instrument.Trumpet, "M-2"),
    "bb-clarinet": lambda: _make(instrument.Clarinet, "M-2"),
    "a-clarinet": lambda: _make(instrument.Clarinet, "m-3"),
    "eb-alto-sax": lambda: _make(instrument.AltoSaxophone, "M-6"),
    "bb-tenor-sax": lambda: _make(instrument.TenorSaxophone, "M-9"),
    "f-horn": lambda: _make(instrument.Horn, "P-5"),
}


def resolve_instrument(name: str) -> instrument.Instrument:
    """Look up a transposing instrument by name (case-insensitive).

    Raises ``ValueError`` naming the valid keys if ``name`` is unknown.
    """
    key = name.strip().lower()
    factory = INSTRUMENTS.get(key)
    if factory is None:
        valid = ", ".join(sorted(INSTRUMENTS))
        raise ValueError(f"unknown instrument {name!r}; valid names: {valid}")
    return factory()
