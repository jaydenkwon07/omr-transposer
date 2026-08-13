from music21 import converter, stream

from omrt.decode.semantic import Token, decode, parse_semantic


def test_parse_semantic_splits_on_whitespace_and_tabs():
    assert parse_semantic("clef-G2\tnote-C4_quarter  barline\t") == [
        "clef-G2", "note-C4_quarter", "barline",
    ]


def test_parse_semantic_blank_is_empty():
    assert parse_semantic("   \t\n") == []


def test_decode_empty_returns_parseable_musicxml():
    xml = decode([])
    parsed = converter.parseData(xml, format="musicxml")
    assert isinstance(parsed, stream.Score)


def test_token_is_str_alias():
    t: Token = "note-C4_quarter"
    assert isinstance(t, str)


def _round_trip(tokens):
    """decode then re-encode with the existing forward encoder."""
    from omrt.eval.symbols import to_symbols
    return to_symbols(decode(tokens))


def test_single_note_round_trips():
    assert _round_trip(["note-C4_quarter"]) == ["note-C4_quarter", "barline"]


def test_flat_and_sharp_pitches_round_trip():
    assert _round_trip(["note-Eb4_eighth", "note-F#3_eighth"]) == [
        "note-Eb4_eighth", "note-F#3_eighth", "barline",
    ]


def test_dotted_and_mapped_durations_round_trip():
    tokens = ["note-C4_eighth.", "note-D4_thirty_second", "note-E4_sixteenth"]
    assert _round_trip(tokens) == tokens + ["barline"]


def test_rest_round_trips():
    assert _round_trip(["rest-quarter", "rest-eighth."]) == ["rest-quarter", "rest-eighth.", "barline"]


def test_barline_terminated_stream_round_trips_exactly():
    # Ends in barline: the empty measure after the final barline must be dropped,
    # so no phantom trailing barline appears.
    tokens = ["note-C4_quarter", "barline", "note-D4_quarter", "barline"]
    assert _round_trip(tokens) == tokens


def test_stream_not_ending_in_barline_gains_one_barline():
    # Documents the ceiling-loss class: the encoder always closes the last measure.
    tokens = ["note-C4_quarter", "barline", "note-D4_quarter"]
    assert _round_trip(tokens) == tokens + ["barline"]


def test_empty_stream_round_trips_to_empty_list():
    # No empty measure is appended, so there is nothing for to_symbols to barline.
    assert _round_trip([]) == []


def test_clef_key_time_round_trip():
    # Ends in barline so the round trip is exact identity (see Task 3 semantics).
    tokens = ["clef-G2", "keySignature-BbM", "timeSignature-C/", "note-D5_half", "barline"]
    assert _round_trip(tokens) == tokens


def test_all_key_signatures_minus7_to_plus7_round_trip():
    for tonic in ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C",
                  "G", "D", "A", "E", "B", "F#", "C#"]:
        tok = f"keySignature-{tonic}M"
        assert _round_trip(["clef-G2", tok, "note-C4_quarter"])[1] == tok


def test_common_and_cut_and_numeric_time_signatures():
    for tok in ["timeSignature-C", "timeSignature-C/", "timeSignature-6/8", "timeSignature-12/8"]:
        assert _round_trip(["clef-G2", tok, "note-C4_quarter"])[1] == tok


def test_mid_stream_clef_change_round_trips():
    tokens = ["clef-F4", "note-G3_eighth", "barline", "clef-C4", "note-C4_eighth", "barline"]
    assert _round_trip(tokens) == tokens


def test_no_phantom_clef_or_meter_on_attribute_less_stream():
    # A stream with no clef/key/time must NOT gain any on export.
    result = _round_trip(["note-C4_quarter"])
    assert not any(t.startswith("clef-") or t.startswith("timeSignature-")
                   or t.startswith("keySignature-") for t in result)


def test_gracenote_round_trips():
    tokens = ["clef-G2", "gracenote-C4_eighth", "note-Bb3_quarter.", "barline"]
    assert _round_trip(tokens) == tokens


def test_multirest_round_trips():
    tokens = ["clef-C1", "multirest-25", "barline", "note-F5_quarter.", "barline"]
    assert _round_trip(tokens) == tokens


def test_fermata_round_trips():
    tokens = ["clef-G2", "note-C4_quarter_fermata", "barline"]
    assert _round_trip(tokens) == tokens


def test_rest_fermata_round_trips():
    tokens = ["clef-G2", "rest-quarter_fermata", "barline"]
    assert _round_trip(tokens) == tokens


def test_two_note_tie_round_trips():
    tokens = ["clef-G2", "note-F5_quarter.", "tie", "note-F5_quarter", "barline"]
    assert _round_trip(tokens) == tokens


def test_three_note_tie_chain_round_trips():
    tokens = ["clef-G2", "note-F5_quarter", "tie", "note-F5_quarter", "tie", "note-F5_quarter", "barline"]
    assert _round_trip(tokens) == tokens


def test_tie_sets_start_continue_stop():
    from omrt.decode.semantic import decode
    from music21 import converter
    tokens = ["note-F5_quarter", "tie", "note-F5_quarter", "tie", "note-F5_quarter"]
    notes = list(converter.parseData(decode(tokens), format="musicxml").recurse().notes)
    assert [n.tie.type for n in notes] == ["start", "continue", "stop"]


def test_unknown_token_is_skipped_not_raised():
    assert _round_trip(["clef-G2", "blah", "note-C4_quarter", "barline"]) == [
        "clef-G2", "note-C4_quarter", "barline",
    ]


def test_malformed_pitch_is_skipped():
    assert _round_trip(["clef-G2", "note-Z9_eighth", "note-C4_quarter", "barline"]) == [
        "clef-G2", "note-C4_quarter", "barline",
    ]


def test_leading_tie_is_dropped():
    assert _round_trip(["tie", "note-C4_quarter", "barline"]) == ["note-C4_quarter", "barline"]


def test_empty_round_trips_to_empty_list():
    assert _round_trip([]) == []


def test_all_unknown_returns_empty_musicxml():
    assert _round_trip(["blah", "xyz"]) == []


def test_tie_across_barline_round_trips():
    # Locks the invariant that last_note is NOT reset at a barline (Task 6).
    tokens = ["note-F5_quarter", "tie", "barline", "note-F5_quarter", "barline"]
    assert _round_trip(tokens) == tokens


def test_decode_dicts_are_exact_inverses_of_symbols():
    from omrt.eval import symbols
    from omrt.decode import semantic

    assert semantic._SHARPS_BY_MAJOR_TONIC == {
        v: k for k, v in symbols._MAJOR_TONIC_BY_SHARPS.items()
    }
    assert semantic._DURATION_TYPES == {
        v: k for k, v in symbols._DURATION_NAMES.items()
    }
    assert semantic._TIME_SYMBOLS == {
        v: k for k, v in symbols._TIME_SYMBOLS.items()
    }
