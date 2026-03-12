import torch
import torch.nn as nn
from models.base import BytePredictor


class GRUBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, h=None):
        y = self.ln1(x)
        y, h_new = self.gru(y, h)
        y = self.ln2(y)
        y = self.ff(y)
        return x + y, h_new


class GRUBytePredictor(BytePredictor):
    def __init__(self, d_model: int = 256, num_layers: int = 4,
                 vocab_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [GRUBlock(d_model) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, x, inference_params=None):
        h = self.embedding(x)
        for blk in self.blocks:
            h, _ = blk(h)
        return self.head(h)

    @torch.inference_mode()
    def init_stream(self, max_len: int, batch_size: int = 1,
                    device=None, dtype=None):
        return [torch.zeros(1, batch_size, self.d_model, device=device)
                for _ in range(self.num_layers)]

    @torch.inference_mode()
    def step(self, byte_t: torch.LongTensor, caches: list) -> torch.Tensor:
        h = self.embedding(byte_t).unsqueeze(1)
        for i, blk in enumerate(self.blocks):
            h, caches[i] = blk(h, caches[i])
        return self.head(h.squeeze(1))
