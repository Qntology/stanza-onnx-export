import torch

def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if not torch.jit.is_tracing():
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
    if not isinstance(sequences, (list, tuple)): return sequences
    max_len = max(s.size(0) for s in sequences)
    padded = [torch.nn.functional.pad(s, (0, 0, 0, max_len - s.size(0)) if s.dim() > 1 else (0, max_len - s.size(0)), value=padding_value) for s in sequences]
    return torch.stack(padded, dim=0 if batch_first else 1)
