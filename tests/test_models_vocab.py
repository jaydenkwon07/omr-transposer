from omrt.models.vocab import Vocabulary


def test_blank_is_index_zero_and_size_counts_it():
    v = Vocabulary.build([["barline", "note-C4_quarter"], ["barline"]])
    assert v.blank_index == 0
    assert v.size == 3  # blank + 2 distinct tokens


def test_encode_decode_round_trip():
    v = Vocabulary.build([["clef-G2", "barline", "clef-G2"]])
    ids = v.encode(["clef-G2", "barline"])
    assert 0 not in ids  # never emits blank
    assert v.decode(ids) == ["clef-G2", "barline"]


def test_save_load_is_stable(tmp_path):
    v = Vocabulary.build([["b", "a", "c"]])
    p = tmp_path / "vocab.json"
    v.save(str(p))
    w = Vocabulary.load(str(p))
    assert w.itos == v.itos
    assert w.encode(["a", "b", "c"]) == v.encode(["a", "b", "c"])
