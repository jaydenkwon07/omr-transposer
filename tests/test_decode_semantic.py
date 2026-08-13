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
