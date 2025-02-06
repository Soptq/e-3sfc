import torch
import math

import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x
    

class TransformerClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_class,
        max_len=128,
        nhead=8,
        dim_feedforward=2048,
        num_layers=6,
    ):
        super().__init__()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.emb = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, num_class)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, text, offsets):
        embedded = self.emb(text) * math.sqrt(self.d_model)
        x = torch.zeros((self.max_len, len(offsets), self.d_model), device=embedded.device)
        for i, offset in enumerate(offsets):
            begin = offset.item()
            end = offsets[i + 1].item() if i + 1 < len(offsets) else text.size(0)
            if end - begin > self.max_len:
                end = begin + self.max_len
            x[0:end - begin, i, :] = embedded[begin:end, :]

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x
    