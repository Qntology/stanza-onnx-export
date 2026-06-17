import stanza
import json

lang = 'ko'
print(f"Extracting Vocab for: {lang}")
# Force kaist package which is more likely to have pretrained
nlp = stanza.Pipeline(lang, processors='tokenize,pos', package={'pos': 'kaist'})
pos_model = nlp.processors['pos'].trainer.model

vocab_data = {
    "word2id": pos_model.vocab['word']._unit2id,
    "id2upos": pos_model.vocab['upos']._id2unit
}

if hasattr(pos_model, 'pretrain') and pos_model.pretrain:
    print(f"  Found pretrain object for {lang}")
    vocab_data["pre2id"] = pos_model.pretrain.vocab._unit2id
    print(f"  Pretrained vocab size: {len(vocab_data['pre2id'])}")

with open(f"vocabs/{lang}_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False)

print("Done.")
