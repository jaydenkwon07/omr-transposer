from __future__ import annotations

import torch

from omrt.datagen.types import Image
from omrt.decode import Token
from omrt.models.crnn import CRNN
from omrt.models.ctc import ctc_greedy_decode
from omrt.models.dataset import preprocess
from omrt.models.vocab import Vocabulary


class CRNNModel:
    """Seam-2 Model: wraps a trained CRNN + its Vocabulary behind `predict`."""

    def __init__(self, model: CRNN, vocab: Vocabulary, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device

    @torch.no_grad()
    def predict(self, image: Image) -> list[Token]:
        x = preprocess(image).unsqueeze(0).to(self.device)  # [1,1,128,W]
        log_probs = self.model(x)                            # [T,1,C]
        input_lengths = torch.tensor([log_probs.shape[0]])
        ids = ctc_greedy_decode(log_probs.cpu(), input_lengths)[0]
        return self.vocab.decode(ids)

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "CRNNModel":
        dev = device or torch.device("cpu")
        ckpt = torch.load(path, map_location=dev)
        vocab = Vocabulary.from_dict(ckpt["vocab"])
        model = CRNN(vocab_size=vocab.size)
        model.load_state_dict(ckpt["model"])
        return cls(model, vocab, dev)
