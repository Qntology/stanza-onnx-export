import stanza
import json
import os

langs = ['en', 'ja', 'zh-hans', 'vi', 'ar', 'es', 'ko']
output_dir = "./vocabs"
os.makedirs(output_dir, exist_ok=True)

for lang in langs:
    print(f"Extracting Vocab for: {lang}")
    try:
        # download_method=None to avoid re-downloading if already present
        nlp = stanza.Pipeline(lang, processors='tokenize,pos')
        pos_model = nlp.processors['pos'].trainer.model
        
        vocab_data = {
            "word2id": pos_model.vocab['word']._unit2id,
            "id2upos": pos_model.vocab['upos']._id2unit
        }
        
        # Check for pretrain object or pretrained in vocab
        pre2id = None
        if hasattr(pos_model, 'pretrain') and pos_model.pretrain is not None:
            print(f"  Found pretrain object for {lang}")
            pre2id = pos_model.pretrain.vocab._unit2id
        elif 'pretrained' in pos_model.vocab:
            print(f"  Found pretrained vocab in pos_model.vocab for {lang}")
            pre2id = pos_model.vocab['pretrained']._unit2id
            
        if pre2id:
            vocab_data["pre2id"] = pre2id
            print(f"  Pretrained vocab size: {len(pre2id)}")
        else:
            print(f"  No pretrained vocab for {lang}")
            
        with open(f"{output_dir}/{lang}_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False)
            
        print(f"✅ Saved {lang} vocab")
    except Exception as e:
        print(f"❌ Failed {lang}: {e}")

print("Vocab extraction finished.")
