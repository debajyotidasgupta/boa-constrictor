import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from models.base import BytePredictor


@dataclass
class TransformerCache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    seq_offset: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 65536):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.max_len = max_len

    def forward(self, x, kv_cache=None, layer_idx=None):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is not None and layer_idx is not None:
            if kv_cache.keys[layer_idx] is not None:
                k = torch.cat([kv_cache.keys[layer_idx], k], dim=2)
                v = torch.cat([kv_cache.values[layer_idx], v], dim=2)
            kv_cache.keys[layer_idx] = k
            kv_cache.values[layer_idx] = v

        T = k.shape[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        q_pos = torch.arange(T - L, T, device=x.device).unsqueeze(1)
        k_pos = torch.arange(T, device=x.device).unsqueeze(0)
        mask = k_pos <= q_pos
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 65536):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, kv_cache=None, layer_idx=None):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.ff(self.ln2(x))
        return x


class TransformerBytePredictor(BytePredictor):
    def __init__(self, d_model: int = 256, num_layers: int = 4,
                 vocab_size: int = 256, n_heads: int = 8,
                 max_len: int = 65536):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_len = max_len
        self.n_heads = n_heads

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_len)
             for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size),
        )

    def _ensure_pos_capacity(self, needed_len: int, device: torch.device):
        cur_len = self.pos_embedding.num_embeddings
        if needed_len <= cur_len:
            return
        new_len = max(needed_len, int(cur_len * 1.5))
        new_emb = nn.Embedding(new_len, self.d_model).to(device)
        with torch.no_grad():
            new_emb.weight[:cur_len].copy_(self.pos_embedding.weight.to(device))
            nn.init.normal_(new_emb.weight[cur_len:], mean=0.0, std=0.02)
        self.pos_embedding = new_emb

    def forward(self, x, inference_params=None):
        B, L = x.shape
        self._ensure_pos_capacity(L, x.device)
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)

    @torch.inference_mode()
    def init_stream(self, max_len: int, batch_size: int = 1,
                    device=None, dtype=None):
        return TransformerCache(
            keys=[None] * self.num_layers,
            values=[None] * self.num_layers,
            seq_offset=0,
        )

    @torch.inference_mode()
    def step(self, byte_t: torch.LongTensor, cache: TransformerCache) -> torch.Tensor:
        self._ensure_pos_capacity(cache.seq_offset + 1, byte_t.device)
        pos = torch.tensor([cache.seq_offset], device=byte_t.device)
        h = self.embedding(byte_t).unsqueeze(1) + self.pos_embedding(pos).unsqueeze(0)
        for i, blk in enumerate(self.blocks):
            h = blk(h, kv_cache=cache, layer_idx=i)
        h = self.ln_f(h).squeeze(1)
        cache.seq_offset += 1
        return self.head(h)
