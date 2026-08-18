from __future__ import annotations

import torch
from torch import Tensor, nn


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, pool: tuple[int, int]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(torch.relu(self.bn(self.conv(x))))  # type: ignore[no-any-return]


class CRNN(nn.Module):
    """Calvo-Zaragoza & Rizo 2018, Table 2. Input [N,1,128,W] -> log-probs [T=W/4, N, vocab_size]."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBlock(1, 32, (2, 2)),
            _ConvBlock(32, 64, (2, 2)),
            _ConvBlock(64, 128, (2, 1)),
            _ConvBlock(128, 256, (2, 1)),
        )
        # height 128 -> 8 after four /2 poolings; features = 256 channels * 8 rows.
        self.rnn1 = nn.LSTM(256 * 8, 256, bidirectional=True, batch_first=False)
        self.rnn2 = nn.LSTM(512, 256, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)                      # [N, 256, 8, W/4]
        n, c, h, w = x.shape
        x = x.permute(3, 0, 1, 2).reshape(w, n, c * h)  # [T, N, 2048]
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return torch.log_softmax(self.fc(x), dim=2)
