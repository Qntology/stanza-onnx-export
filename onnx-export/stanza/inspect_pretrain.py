import torch

path = "ko_default/pretrain/conll17.pt"
checkpoint = torch.load(path, map_location='cpu')
print(f"Keys: {checkpoint.keys()}")
for k in checkpoint.keys():
    if not isinstance(checkpoint[k], (torch.Tensor, dict)):
        print(f"  {k}: {checkpoint[k]}")
    elif isinstance(checkpoint[k], dict):
        print(f"  {k} (dict) keys: {checkpoint[k].keys()}")
