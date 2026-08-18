import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "make_split", os.path.join("scripts", "make_split.py"))
make_split_mod = importlib.util.module_from_spec(_spec)


def _load():
    _spec.loader.exec_module(make_split_mod)
    return make_split_mod.make_split


def test_split_is_deterministic_disjoint_and_proportioned():
    make_split = _load()
    ids = [f"id{i}" for i in range(1000)]
    a = make_split(ids, seed=0)
    b = make_split(ids, seed=0)
    assert a == b
    assert len(a["train"]) == 800 and len(a["val"]) == 100 and len(a["test"]) == 100
    everything = set(a["train"]) | set(a["val"]) | set(a["test"])
    assert everything == set(ids)
    assert not (set(a["train"]) & set(a["val"]))
    assert not (set(a["train"]) & set(a["test"]))
    assert not (set(a["val"]) & set(a["test"]))
