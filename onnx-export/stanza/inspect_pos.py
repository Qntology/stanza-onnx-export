import stanza
import torch

model_dir = './temp_models'
nlp = stanza.Pipeline(lang='ko', dir=model_dir, processors='tokenize,pos,lemma,depparse', verbose=True, use_gpu=False)
pos_model = nlp.processors['pos'].trainer.model

print(f"Args charlm: {pos_model.args.get('charlm')}")
print(f"Bert model: {pos_model.args.get('bert_model')}")
