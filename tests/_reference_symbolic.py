"""Minimal REFERENCE implementation, used only to validate the test suite."""
from music21 import converter, interval, key, note, stream, instrument
from music21.musicxml.m21ToXml import GeneralObjectExporter

_INSTRUMENTS = {
    "bb-trumpet": lambda: _mk(instrument.Trumpet(), "M-2"),
    "bb-clarinet": lambda: _mk(instrument.Clarinet(), "M-2"),
    "eb-alto-sax": lambda: _mk(instrument.AltoSaxophone(), "M-6"),
    "f-horn": lambda: _mk(instrument.Horn(), "P-5"),
}

def _mk(inst, iv):
    inst.transposition = interval.Interval(iv)
    return inst

def _xml(sc): return GeneralObjectExporter().parse(sc).decode("utf-8")


def _fix_accidentals(sc):
    for p in sc.parts:                       # Part level, NOT Score level
        p.makeAccidentals(inPlace=True, overrideStatus=True)
    return sc

def transpose_by_interval(xml: str, iv: str) -> str:
    sc = converter.parse(xml)
    out = sc.transpose(interval.Interval(iv))
    return _xml(_fix_accidentals(out))

def _first_key(sc):
    ks = list(sc.recurse().getElementsByClass(key.KeySignature))
    if not ks:
        raise ValueError("no explicit key signature")
    k = ks[0]
    return k.asKey() if not isinstance(k, key.Key) else k

def transpose_to_key(xml: str, target: str) -> str:
    sc = converter.parse(xml)
    src = _first_key(sc)
    tgt = key.Key(*target.split())
    up = interval.Interval(noteStart=note.Note(src.tonic.name + "4"),
                           noteEnd=note.Note(tgt.tonic.name + "4"))
    # choose the smaller motion; tie goes downward
    down = up.complement.reverse() if up.semitones else up
    iv = up if abs(up.semitones) <= 6 else interval.Interval(up.semitones - 12)
    if abs(up.semitones) > 6:
        iv = interval.Interval(noteStart=note.Note(src.tonic.name + "4"),
                               noteEnd=note.Note(tgt.tonic.name + "3"))
    out = sc.transpose(iv)
    return _xml(_fix_accidentals(out))

def transpose_for_instrument(xml: str, name: str) -> str:
    if name not in _INSTRUMENTS:
        raise ValueError(f"unknown instrument {name!r}; valid: {sorted(_INSTRUMENTS)}")
    sc = converter.parse(xml)
    sc = sc.toSoundingPitch()            # normalize FIRST
    for p in sc.parts:
        for old in list(p.getElementsByClass(instrument.Instrument)):
            p.remove(old)
        p.insert(0, _INSTRUMENTS[name]())
    sc = sc.toWrittenPitch()
    return _xml(sc)
