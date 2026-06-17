import torch
import json

path = "ko_default/pos/kaist_nocharlm.pt"
print(f"Loading {path}")
checkpoint = torch.load(path, map_location='cpu')

if 'vocab' in checkpoint:
    vocab = checkpoint['vocab']
    print(f"Vocab keys: {vocab.keys()}")
    
    # Extract mappings
    data = {
        "word2id": vocab['word'].get('_unit2id', {}),
        "id2upos": vocab['upos'].get('_id2unit', []),
    }
    if 'char' in vocab:
        data["char2id"] = vocab['char'].get('_unit2id', {})
    if 'pretrained' in vocab:
        data["pre2id"] = vocab['pretrained'].get('_unit2id', {})
        
    with open("vocabs/ko_vocab.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("✅ Saved ko_vocab.json from checkpoint")
else:
    print("No vocab in checkpoint")
