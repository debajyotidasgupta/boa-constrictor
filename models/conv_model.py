import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from models.base import BytePredictor


@dataclass
class ConvCache:
    buffers: list[torch.Tensor] = field(default_factory=list)
    write_pos: list[int] = field(default_factory=list)


class CausalConv1dBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation

        self.ln = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, 2 * d_model, kernel_size,
                              dilation=dilation)
        self.proj = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        residual = x
        y = self.ln(x.transpose(1, 2)).transpose(1, 2)
        y = F.pad(y, (self.pad, 0))
        y = self.conv(y)
        gate, filt = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.proj(y)
        return residual + y

    def init_cache(self, batch_size: int, d_model: int, device):
        buf_len = self.pad
        buf = torch.zeros(batch_size, d_model, buf_len, device=device)
        return buf

    def step_cached(self, x_col, buf):
        residual = x_col
        y = self.ln(x_col.transpose(1, 2)).transpose(1, 2)

        full = torch.cat([buf, y], dim=2)
        new_buf = full[:, :, 1:]
        y = self.conv(full)
        gate, filt = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.proj(y)
        return residual + y, new_buf


class ConvBytePredictor(BytePredictor):
    def __init__(self, d_model: int = 256, num_layers: int = 4,
                 vocab_size: int = 256, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 8)
            self.blocks.append(
                CausalConv1dBlock(d_model, kernel_size, dilation)
            )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, x, inference_params=None):
        h = self.embedding(x)
        h = h.transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2)
        h = self.ln_f(h)
        return self.head(h)

    @torch.inference_mode()
    def init_stream(self, max_len: int, batch_size: int = 1,
                    device=None, dtype=None):
        cache = ConvCache()
        for blk in self.blocks:
            cache.buffers.append(
                blk.init_cache(batch_size, self.d_model, device)
            )
        return cache

    @torch.inference_mode()
    def step(self, byte_t: torch.LongTensor, cache: ConvCache) -> torch.Tensor:
        h = self.embedding(byte_t).unsqueeze(2)
        for i, blk in enumerate(self.blocks):
            h, cache.buffers[i] = blk.step_cached(h, cache.buffers[i])
        h = h.squeeze(2)
        h = self.ln_f(h)
        return self.head(h)
