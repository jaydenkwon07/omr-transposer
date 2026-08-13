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
