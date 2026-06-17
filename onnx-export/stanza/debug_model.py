import stanza
import torch

lang = 'en'
nlp = stanza.Pipeline(lang, processors='tokenize,pos')
pos_model = nlp.processors['pos'].trainer.model

print(f"Model keys: {pos_model.__dict__.keys()}")
if hasattr(pos_model, 'vocab'):
    print(f"Vocab keys: {pos_model.vocab.keys()}")
    for k, v in pos_model.vocab.items():
        print(f"  {k}: {type(v)}")

if hasattr(pos_model, 'pretrain') and pos_model.pretrain:
    print(f"Pretrain object: {pos_model.pretrain}")
    print(f"Pretrain vocab size: {len(pos_model.pretrain.vocab)}")
