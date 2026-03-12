import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any


class BytePredictor(nn.Module, ABC):
    embedding: nn.Embedding

    @abstractmethod
    def forward(self, x: torch.LongTensor, inference_params: Any = None) -> torch.Tensor:
        ...

    @abstractmethod
    @torch.inference_mode()
    def init_stream(self, max_len: int, batch_size: int = 1,
                    device=None, dtype=None) -> Any:
        ...

    @abstractmethod
    @torch.inference_mode()
    def step(self, byte_t: torch.LongTensor, cache: Any) -> torch.Tensor:
        ...

    @property
    def vocab_size(self) -> int:
        return self.embedding.num_embeddings

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def memory_footprint_mb(self) -> float:
        total_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        return total_bytes / (1024 * 1024)
