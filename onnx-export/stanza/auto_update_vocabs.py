import stanza
import os
import torch
import json

def update_vocab(lang, pretrain_path):
    print(f"Updating {lang} with {pretrain_path}")
    checkpoint = torch.load(pretrain_path, map_location='cpu')
    if 'vocab' in checkpoint and isinstance(checkpoint['vocab'], dict) and '_unit2id' in checkpoint['vocab']:
        unit2id = checkpoint['vocab']['_unit2id']
        print(f"  Pretrained vocab size: {len(unit2id)}")
        
        vocab_path = f"vocabs/{lang}_vocab.json"
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["pre2id"] = unit2id
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"  ✅ Updated {vocab_path}")
        else:
            print(f"  ❌ {vocab_path} not found")

# Try to find common paths
stanza_dir = os.path.join(os.getenv('USERPROFILE'), 'stanza_resources') # Default
if not os.path.exists(stanza_dir):
    stanza_dir = os.path.join(os.getenv('APPDATA'), 'StanfordNLP', 'stanza', 'Cache') # Another common path

print(f"Searching in {stanza_dir}")

# For English
en_pretrain = os.path.join(os.getenv('USERPROFILE'), 'stanza_resources', 'en', 'pretrain', 'combined.pt')
# If not found, try to find any .pt in en/pretrain
if not os.path.exists(en_pretrain):
    # Search recursively
    for root, dirs, files in os.walk(stanza_dir):
        if 'en' in root and 'pretrain' in root:
            for f in files:
                if f.endswith('.pt'):
                    en_pretrain = os.path.join(root, f)
                    update_vocab('en', en_pretrain)
                    break

# For others...
# Actually, I'll just search for all languages I have in vocabs/
langs = ['en', 'ja', 'zh-hans', 'vi', 'ar', 'es']
for lang in langs:
    found = False
    for root, dirs, files in os.walk(stanza_dir):
        if lang in root and 'pretrain' in root:
            for f in files:
                if f.endswith('.pt'):
                    update_vocab(lang, os.path.join(root, f))
                    found = True
                    break
            if found: break
