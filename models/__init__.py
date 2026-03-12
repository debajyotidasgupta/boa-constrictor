from models.base import BytePredictor
from models.mamba_model import MambaBytePredictor, MambaByteModel
from models.transformer_model import TransformerBytePredictor
from models.gru_model import GRUBytePredictor
from models.conv_model import ConvBytePredictor

MODEL_REGISTRY: dict[str, type[BytePredictor]] = {
    "mamba": MambaBytePredictor,
    "transformer": TransformerBytePredictor,
    "gru": GRUBytePredictor,
    "conv": ConvBytePredictor,
}


def create_model(arch: str, d_model: int = 256, num_layers: int = 4,
                 vocab_size: int = 256, device: str = "cuda",
                 **kwargs) -> BytePredictor:
    if arch not in MODEL_REGISTRY:
        avail = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown architecture '{arch}'. Available: {avail}")
    cls = MODEL_REGISTRY[arch]
    model = cls(d_model=d_model, num_layers=num_layers,
                vocab_size=vocab_size, **kwargs)
    if device != "cpu":
        model = model.to(device)
    return model


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


__all__ = [
    "BytePredictor",
    "MambaBytePredictor",
    "TransformerBytePredictor",
    "GRUBytePredictor",
    "ConvBytePredictor",
    "MODEL_REGISTRY",
    "create_model",
    "list_models",
]
