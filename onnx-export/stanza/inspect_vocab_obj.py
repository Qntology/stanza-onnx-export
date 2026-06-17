import torch

path = "ko_default/pos/kaist_nocharlm.pt"
checkpoint = torch.load(path, map_location='cpu')
vocab = checkpoint['vocab']

print(f"Type of vocab['word']: {type(vocab['word'])}")
if isinstance(vocab['word'], dict):
    print(f"Keys: {vocab['word'].keys()}")
else:
    print(f"Attributes: {dir(vocab['word'])}")
