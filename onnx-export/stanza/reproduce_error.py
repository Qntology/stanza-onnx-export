import stanza
import torch
import torch.nn as nn
import os

class POSWrapper(nn.Module):
    def __init__(self, model, input_names):
        super().__init__()
        self.model = model
        self.input_names = input_names
    def forward(self, *args):
        kwargs = {name: arg for name, arg in zip(self.input_names, args)}
        def g(name): return kwargs.get(name)
        
        # Stanza POS forward: 
        # (word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text)
        return self.model(
            g('word'), g('word_mask'), g('wordchars'), g('wordchars_mask'),
            g('upos'), g('xpos'), g('ufeats'), g('pretrained'),
            g('word_orig_idx'), g('sentlens'), g('wordlens'), None
        )

lang = 'sq'
stanza.download(lang, processors='tokenize,pos')
nlp = stanza.Pipeline(lang, processors='tokenize,pos')
pos_model = nlp.processors['pos'].trainer.model
pos_model.cpu().eval()

batch_size, seq_len = 1, 5
pos_input_names = ['word', 'word_mask', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens']
pos_inputs = [
    torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long),
    torch.ones((batch_size, seq_len), dtype=torch.bool),
    torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long),
    torch.arange(batch_size * seq_len, dtype=torch.long),
    torch.tensor([seq_len] * batch_size, dtype=torch.long),
    torch.ones((batch_size * seq_len,), dtype=torch.long)
]

if hasattr(pos_model, 'charmodel') and pos_model.charmodel:
    pos_input_names += ['wordchars', 'wordchars_mask']
    pos_inputs += [
        torch.randint(0, 10, (batch_size * seq_len, 10), dtype=torch.long),
        torch.ones((batch_size * seq_len, 10), dtype=torch.bool)
    ]

pos_wrapper = POSWrapper(pos_model, pos_input_names)

print("Attempting export...")
try:
    torch.onnx.export(
        pos_wrapper, tuple(pos_inputs), "test_sq_pos.onnx",
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=pos_input_names
    )
    print("Export Success!")
except Exception as e:
    print(f"Export Failed: {e}")
    import traceback
    traceback.print_exc()
