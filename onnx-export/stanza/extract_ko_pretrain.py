import torch
import json

path = "ko_default/pretrain/conll17.pt"
print(f"Loading {path}")
checkpoint = torch.load(path, map_location='cpu')

# Pretrain checkpoints in Stanza have a 'vocab' attribute
if 'vocab' in checkpoint:
    vocab = checkpoint['vocab']
    if isinstance(vocab, dict) and '_unit2id' in vocab:
        unit2id = vocab['_unit2id']
    else:
        unit2id = {}
        
    print(f"Pretrained vocab size: {len(unit2id)}")
    
    # Load current ko_vocab and update it
    with open("vocabs/ko_vocab.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data["pre2id"] = unit2id
    
    with open("vocabs/ko_vocab.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("✅ Updated ko_vocab.json with pretrained mapping")
else:
    print("No vocab found in pretrain checkpoint")
    print(f"Keys: {checkpoint.keys()}")
