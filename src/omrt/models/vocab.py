from __future__ import annotations

import json
from typing import Iterable


class Vocabulary:
    """Maps semantic tokens to CTC class indices. Index 0 is reserved for the CTC blank;
    real tokens occupy 1..N. `itos` excludes the blank; `size` includes it."""

    blank_index = 0

    def __init__(self, itos: list[str]) -> None:
        self.itos = list(itos)
        self._stoi = {tok: i + 1 for i, tok in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists: Iterable[list[str]]) -> Vocabulary:
        seen: set[str] = set()
        for toks in token_lists:
            seen.update(toks)
        return cls(sorted(seen))

    @property
    def size(self) -> int:
        return len(self.itos) + 1

    def encode(self, tokens: list[str]) -> list[int]:
        return [self._stoi[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i - 1] for i in ids if i != self.blank_index]

    def to_dict(self) -> dict[str, object]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Vocabulary:
        itos = d["itos"]
        assert isinstance(itos, list)
        return cls([str(t) for t in itos])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path: str) -> Vocabulary:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
