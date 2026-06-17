import stanza
import json

lang = 'en'
print(f"Extracting Vocab for: {lang}")
nlp = stanza.Pipeline(lang, processors='tokenize,pos')
pos_model = nlp.processors['pos'].trainer.model

vocab_data = {
    "word2id": pos_model.vocab['word']._unit2id,
    "id2upos": pos_model.vocab['upos']._id2unit
}

if hasattr(pos_model, 'pretrain') and pos_model.pretrain is not None:
    print(f"  Found pretrain object for {lang}")
    vocab_data["pre2id"] = pos_model.pretrain.vocab._unit2id
    print(f"  Pretrained vocab size: {len(vocab_data['pre2id'])}")
elif 'pretrained' in pos_model.vocab:
    print(f"  Found pretrained vocab in pos_model.vocab for {lang}")
    vocab_data["pre2id"] = pos_model.vocab['pretrained']._unit2id
    print(f"  Pretrained vocab size: {len(vocab_data['pre2id'])}")
else:
    print(f"  No pretrained vocab found for {lang}")

with open(f"{lang}_vocab_full.json", "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False)

print("Done.")
