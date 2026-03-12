import torch
import torch.nn as nn
from models.base import BytePredictor

IS_CUDA = torch.cuda.is_available()

if IS_CUDA:
    from mamba_ssm import Mamba
    from mamba_ssm.utils.generation import InferenceParams
else:
    from mambapy.mamba import MambaBlock as MambaCPU, MambaConfig


def _bump_offset(inf, k: int = 1):
    if hasattr(inf, "seqlen_offset"):
        inf.seqlen_offset += k
    elif hasattr(inf, "sequence_length_offset"):
        setattr(inf, "sequence_length_offset",
                getattr(inf, "sequence_length_offset") + k)
    else:
        setattr(inf, "seqlen_offset",
                getattr(inf, "seqlen_offset", 0) + k)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        if IS_CUDA:
            self.mamba = Mamba(d_model=d_model)
        else:
            config = MambaConfig(d_model=d_model, n_layers=0, use_cuda=False)
            self.mamba = MambaCPU(config)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, inference_params=None):
        y = self.ln1(x)
        if IS_CUDA:
            y = self.mamba(y, inference_params=inference_params)
        else:
            y = self.mamba(y)
        y = self.ln2(y)
        y = self.ff(y)
        return x + y

    if not IS_CUDA:
        def init_cache(self, batch_size: int, device):
            d_inner = self.mamba.config.d_inner
            d_conv = self.mamba.config.d_conv
            inputs = torch.zeros(batch_size, d_inner, d_conv - 1, device=device)
            return (None, inputs)

        def step(self, x, cache):
            y = self.ln1(x)
            y, cache = self.mamba.step(y, cache)
            y = self.ln2(y)
            y = self.ff(y)
            return x + y, cache


def _tag_mamba_layers_with_ids(model):
    i = 0
    for m in model.modules():
        if IS_CUDA:
            if isinstance(m, Mamba):
                setattr(m, "layer_idx", i)
                i += 1
        else:
            if isinstance(m, MambaCPU):
                setattr(m, "layer_idx", i)
                i += 1


class MambaBytePredictor(BytePredictor):
    def __init__(self, d_model: int = 256, num_layers: int = 4,
                 vocab_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size),
        )
        _tag_mamba_layers_with_ids(self)

    def forward(self, x, inference_params=None):
        h = self.embedding(x)
        for blk in self.blocks:
            h = blk(h, inference_params=inference_params)
        return self.head(h)

    if IS_CUDA:
        @torch.inference_mode()
        def init_stream(self, max_len: int, batch_size: int = 1,
                        device=None, dtype=None):
            return InferenceParams(max_batch_size=batch_size,
                                   max_seqlen=max_len)

        @torch.inference_mode()
        def step(self, byte_t: torch.LongTensor, inf) -> torch.Tensor:
            x = self.embedding(byte_t).unsqueeze(1)
            h = x
            for blk in self.blocks:
                h = blk(h, inference_params=inf)
            logits_next = self.head(h).squeeze(1)
            _bump_offset(inf, 1)
            return logits_next
    else:
        @torch.inference_mode()
        def init_stream(self, max_len: int, batch_size: int = 1,
                        device=None, dtype=None):
            return [blk.init_cache(batch_size, device) for blk in self.blocks]

        @torch.inference_mode()
        def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
            h = self.embedding(byte_t)
            for i, blk in enumerate(self.blocks):
                h, caches[i] = blk.step(h, caches[i])
            return self.head(h)


def MambaByteModel(d_model=256, num_layers=4, vocab_size=256, device="cuda"):
    return MambaBytePredictor(d_model=d_model, num_layers=num_layers,
                              vocab_size=vocab_size)
