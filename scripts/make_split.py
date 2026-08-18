from __future__ import annotations

import argparse
import json
import os
import random


def make_split(ids: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


if __name__ == "__main__":
    from omrt.models.dataset import list_incipit_ids

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/primus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/primus/split.json")
    args = ap.parse_args()
    split = make_split(list_incipit_ids(args.root), args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(split, fh)
    print({k: len(v) for k, v in split.items()})
