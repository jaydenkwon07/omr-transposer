"""Seam 4: symbol error rate over MusicXML.

The metric is built and tested before any decoder or model exists — a buggy metric
reports improvement that isn't there.
"""
from __future__ import annotations

import pytest
from music21 import clef, expressions, key, meter, note, spanner, stream, tie
from music21.musicxml.m21ToXml import GeneralObjectExporter

from hypothesis import given, settings
from hypothesis import strategies as st

from omrt.eval import Metrics, evaluate, levenshtein, to_symbols
from helpers import make_score


def _xml(score: stream.Score) -> str:
    return GeneralObjectExporter(score).parse().decode("utf-8")


def _one_part(*elements: object) -> str:
    """Wrap elements in a single measure of a single-staff part, as MusicXML."""
    measure = stream.Measure(number=1)
    for element in elements:
        measure.append(element)
    part = stream.Part()
    part.append(measure)
    score = stream.Score()
    score.insert(0, part)
    return _xml(score)


def test_levenshtein_identity_is_zero() -> None:
    ops = levenshtein(["a", "b", "c"], ["a", "b", "c"])
    assert ops.distance == 0
    assert (ops.substitutions, ops.insertions, ops.deletions) == (0, 0, 0)


def test_levenshtein_counts_substitutions() -> None:
    ops = levenshtein(["a", "b", "c"], ["a", "X", "c"])
    assert ops.distance == 1
    assert (ops.substitutions, ops.insertions, ops.deletions) == (1, 0, 0)


def test_levenshtein_counts_insertions() -> None:
    # b (the reference) has one symbol a does not: reaching b requires an insertion.
    ops = levenshtein(["a", "c"], ["a", "b", "c"])
    assert ops.distance == 1
    assert (ops.substitutions, ops.insertions, ops.deletions) == (0, 1, 0)


def test_levenshtein_counts_deletions() -> None:
    ops = levenshtein(["a", "b", "c"], ["a", "c"])
    assert ops.distance == 1
    assert (ops.substitutions, ops.insertions, ops.deletions) == (0, 0, 1)


def test_levenshtein_empty_against_nonempty_is_all_insertions() -> None:
    ops = levenshtein([], ["a", "b"])
    assert ops.distance == 2
    assert (ops.substitutions, ops.insertions, ops.deletions) == (0, 2, 0)


def test_levenshtein_op_counts_sum_to_distance() -> None:
    ops = levenshtein(["a", "b", "c", "d"], ["a", "X", "c", "d", "e", "f"])
    assert ops.substitutions + ops.insertions + ops.deletions == ops.distance


# --- to_symbols: the PrIMuS-mirroring grammar -----------------------------------


def test_blank_input_is_empty_list() -> None:
    assert to_symbols("") == []


def test_unparseable_input_is_empty_list() -> None:
    assert to_symbols("<not-musicxml/>") == []


def test_clef_is_sign_and_line() -> None:
    symbols = to_symbols(_one_part(clef.SopranoClef(), note.Note("C4")))
    assert symbols[0] == "clef-C1"


def test_key_signature_named_by_major_equivalent() -> None:
    # Three flats in a minor-mode score. PrIMuS names the *signature*, never the key,
    # so this is EbM and not Cm — deriving it from detect_key would be a systematic
    # mismatch on every minor incipit.
    symbols = to_symbols(_one_part(clef.TrebleClef(), key.Key("c", "minor"), note.Note("C4")))
    assert "keySignature-EbM" in symbols
    assert not any(s.startswith("keySignature-C") for s in symbols)


def test_key_signature_of_c_major_is_emitted_as_cm() -> None:
    symbols = to_symbols(_one_part(clef.TrebleClef(), key.KeySignature(0), note.Note("C4")))
    assert "keySignature-CM" in symbols


def test_common_time_symbol_is_not_a_ratio() -> None:
    ts = meter.TimeSignature("4/4")
    ts.symbol = "common"
    symbols = to_symbols(_one_part(clef.TrebleClef(), ts, note.Note("C4")))
    assert "timeSignature-C" in symbols
    assert "timeSignature-4/4" not in symbols


def test_cut_time_symbol() -> None:
    ts = meter.TimeSignature("2/2")
    ts.symbol = "cut"
    symbols = to_symbols(_one_part(clef.TrebleClef(), ts, note.Note("C4")))
    assert "timeSignature-C/" in symbols


def test_plain_time_signature_is_a_ratio() -> None:
    symbols = to_symbols(_one_part(clef.TrebleClef(), meter.TimeSignature("6/8"), note.Note("C4")))
    assert "timeSignature-6/8" in symbols


def test_note_carries_pitch_accidental_octave_and_duration() -> None:
    symbols = to_symbols(_one_part(note.Note("B-4", quarterLength=0.5)))
    assert "note-Bb4_eighth" in symbols


def test_sharp_is_spelled_with_hash() -> None:
    symbols = to_symbols(_one_part(note.Note("F#5", quarterLength=0.25)))
    assert "note-F#5_sixteenth" in symbols


def test_dotted_duration_gets_a_trailing_dot() -> None:
    symbols = to_symbols(_one_part(note.Note("C4", quarterLength=1.5)))
    assert "note-C4_quarter." in symbols


def test_long_and_short_durations_use_primus_names() -> None:
    symbols = to_symbols(_one_part(note.Note("C4", quarterLength=0.125)))
    assert "note-C4_thirty_second" in symbols


def test_rest_has_duration_only() -> None:
    symbols = to_symbols(_one_part(note.Rest(quarterLength=1.0)))
    assert "rest-quarter" in symbols


def test_fermata_fuses_into_the_token() -> None:
    n = note.Note("A4", quarterLength=2.0)
    n.expressions.append(expressions.Fermata())
    symbols = to_symbols(_one_part(n))
    assert "note-A4_half_fermata" in symbols


def test_gracenote_is_its_own_class() -> None:
    grace = note.Note("F#5", quarterLength=0.5).getGrace()
    symbols = to_symbols(_one_part(note.Note("D5", quarterLength=1.0), grace))
    assert "gracenote-F#5_eighth" in symbols
    assert "note-F#5_eighth" not in symbols


def test_tie_follows_the_note_it_ties_from() -> None:
    tied = note.Note("D5", quarterLength=1.0)
    tied.tie = tie.Tie("start")
    after = note.Note("D5", quarterLength=1.0)
    after.tie = tie.Tie("stop")
    symbols = to_symbols(_one_part(tied, after))
    assert symbols == ["note-D5_quarter", "tie", "note-D5_quarter", "barline"]


def test_multirest_is_one_token_carrying_its_count() -> None:
    part = stream.Part()
    rests = []
    for number in (1, 2, 3):
        measure = stream.Measure(number=number)
        rest = note.Rest(quarterLength=4.0)
        rests.append(rest)
        measure.append(rest)
        part.append(measure)
    part.insert(0, spanner.MultiMeasureRest(rests))
    score = stream.Score()
    score.insert(0, part)

    symbols = to_symbols(_xml(score))

    assert symbols.count("multirest-3") == 1
    assert not any(s.startswith("rest-") for s in symbols)


def test_barline_follows_each_measure() -> None:
    part = stream.Part()
    for number, name in ((1, "C4"), (2, "D4")):
        measure = stream.Measure(number=number)
        measure.append(note.Note(name, quarterLength=4.0))
        part.append(measure)
    score = stream.Score()
    score.insert(0, part)

    assert to_symbols(_xml(score)) == [
        "note-C4_whole",
        "barline",
        "note-D4_whole",
        "barline",
    ]


def test_clef_and_key_repeat_only_when_they_change() -> None:
    part = stream.Part()
    first = stream.Measure(number=1)
    first.append(clef.TrebleClef())
    first.append(key.KeySignature(1))
    first.append(note.Note("C4", quarterLength=4.0))
    second = stream.Measure(number=2)
    second.append(note.Note("D4", quarterLength=4.0))
    third = stream.Measure(number=3)
    third.append(key.KeySignature(-1))
    third.append(note.Note("E4", quarterLength=4.0))
    part.append([first, second, third])
    score = stream.Score()
    score.insert(0, part)

    symbols = to_symbols(_xml(score))

    assert symbols.count("clef-G2") == 1
    assert symbols.count("keySignature-GM") == 1
    assert symbols.count("keySignature-FM") == 1


# --- evaluate: symbol error rate -------------------------------------------------


EIGHT = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]


def test_identity_is_zero() -> None:
    score = make_score(EIGHT)
    metrics = evaluate(score, score)
    assert metrics.ser == 0.0
    assert metrics.edits == 0


def test_empty_prediction_is_one() -> None:
    assert evaluate("", make_score(EIGHT)).ser == 1.0


def test_both_empty_is_zero() -> None:
    metrics = evaluate("", "")
    assert metrics.ser == 0.0
    assert metrics.ref_len == 0


def test_raw_distance_is_symmetric_but_ser_is_not() -> None:
    short = make_score(["C4", "D4"])
    long = make_score(EIGHT)

    forward, backward = evaluate(short, long), evaluate(long, short)

    assert forward.edits == backward.edits
    # SER normalizes by the *truth* length, which differs between the two calls.
    assert forward.ser != backward.ser


def test_three_substitutions_hand_computed() -> None:
    truth = make_score(EIGHT)
    predicted = make_score(["C4", "D#4", "E4", "F#4", "G4", "A4", "B-4", "C5"])

    metrics = evaluate(predicted, truth)

    assert metrics.substitutions == 3
    assert metrics.insertions == 0
    assert metrics.deletions == 0
    # clef + key + time + 8 notes + a barline per 4/4 measure.
    assert metrics.ref_len == 13
    assert metrics.ser == pytest.approx(3 / 13)


def test_spurious_double_accidental_does_not_count() -> None:
    # normalize_spelling collapses a double accidental that is not diatonic to the key,
    # so B##4 and C#5 are the same symbol and the model is not punished for either.
    truth = make_score(["C4", "C#5"])
    predicted = make_score(["C4", "B##4"])

    assert evaluate(predicted, truth).ser == 0.0


def test_metrics_op_counts_sum_to_edits() -> None:
    metrics = evaluate(make_score(["C4", "D4", "E4"]), make_score(EIGHT))
    assert metrics.substitutions + metrics.insertions + metrics.deletions == metrics.edits


@given(
    st.lists(
        st.sampled_from(["C4", "D4", "E4", "F#4", "G4", "A-4", "B4", "C5"]),
        min_size=1,
        max_size=6,
    )
)
@settings(max_examples=25, deadline=None)
def test_self_comparison_is_always_zero(pitches: list[str]) -> None:
    score = make_score(pitches)
    metrics = evaluate(score, score)
    assert metrics.ser == 0.0
    assert metrics.ser >= 0.0
