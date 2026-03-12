import torch
import numpy as np

from models import create_model
from models.mamba_model import MambaBytePredictor, MambaByteModel

def BoaConstrictor(d_model=256, num_layers=4, vocab_size=256, device="cuda"):
    return create_model("mamba", d_model=d_model, num_layers=num_layers,
                        vocab_size=vocab_size, device=device)


def _aligned_len(n_bytes: int, seq_len: int, batch_size: int) -> int:
    block = seq_len * batch_size
    return (n_bytes // block) * block

def make_splits(data_bytes: bytes | np.ndarray, seq_len: int, batch_size: int,
                splits=(0.8, 0.1, 0.1)):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    buf = np.frombuffer(bytes(data_bytes), dtype=np.uint8)
    usable = _aligned_len(len(buf), seq_len, batch_size)
    buf = buf[:usable]

    n = len(buf)
    n_train = _aligned_len(int(n * splits[0]), seq_len, batch_size)
    n_val   = _aligned_len(int(n * splits[1]), seq_len, batch_size)
    n_test  = _aligned_len(n - n_train - n_val, seq_len, batch_size)

    i0, i1, i2 = 0, n_train, n_train + n_val
    train_bytes = buf[i0:i1].tobytes()
    val_bytes   = buf[i1:i2].tobytes()
    test_bytes  = buf[i2:i2+n_test].tobytes()

    return train_bytes, val_bytes, test_bytes

class ByteDataloader:
    """ Simple dataloader that yields batches of bytes. """
    def __init__(self, data_bytes, seq_len=1048576, batch_size=1, device="cuda"):
        self.data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pos = 0
        self.device = device
    def __len__(self):
        """ Returns the total number of batches in the dataset. """
        return len(self.data_bytes) // (self.seq_len * self.batch_size)
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos + self.seq_len * self.batch_size > len(self.data_bytes):
            self.pos = 0
            raise StopIteration
        
        batch_indices = np.arange(self.pos, self.pos + self.seq_len * self.batch_size)
        batch_indices = batch_indices.reshape(self.batch_size, self.seq_len)
        self.pos += self.seq_len * self.batch_size
        
        batch = self.data_bytes[batch_indices]
        return torch.tensor(batch, dtype=torch.long).to(self.device)
