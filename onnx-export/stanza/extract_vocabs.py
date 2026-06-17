import stanza
import json
import torch
import os

langs = ['en', 'ja', 'zh-hans', 'vi', 'ar', 'es']
output_dir = "./vocabs"
os.makedirs(output_dir, exist_ok=True)

for lang in langs:
    print(f"Extracting Vocab for: {lang}")
    try:
        nlp = stanza.Pipeline(lang, processors='tokenize,pos', download_method=None)
        pos_model = nlp.processors['pos'].trainer.model
        
        vocab_data = {
            "word2id": pos_model.vocab['word']._unit2id,
            "id2upos": pos_model.vocab['upos']._id2unit
        }
        
        # Extract pretrained vocab if it exists
        if 'pretrained' in pos_model.vocab:
            print(f"  Found pretrained vocab for {lang}")
            vocab_data["pre2id"] = pos_model.vocab['pretrained']._unit2id
        else:
            print(f"  No pretrained vocab for {lang}")
            
        with open(f"{output_dir}/{lang}_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Saved {lang} vocab")
    except Exception as e:
        print(f"❌ Failed {lang}: {e}")

print("Vocab extraction finished.")
