import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from omrt.models import CRNN, CRNNModel, Model, Vocabulary


def test_model_protocol_is_runtime_checkable():
    class Dummy:
        def predict(self, image):
            return ["barline"]

    assert isinstance(Dummy(), Model)

    class NotAModel:
        pass

    assert not isinstance(NotAModel(), Model)


def test_predict_returns_vocab_tokens():
    vocab = Vocabulary.build([["barline", "clef-G2", "note-C4_quarter"]])
    model = CRNN(vocab_size=vocab.size)
    wrapped = CRNNModel(model, vocab, torch.device("cpu"))
    image = np.full((100, 400), 255, dtype=np.uint8)
    out = wrapped.predict(image)
    assert isinstance(out, list)
    assert all(tok in vocab.itos for tok in out)  # untrained: whatever it emits is valid vocab
    assert isinstance(wrapped, __import__("omrt.models", fromlist=["Model"]).Model)
