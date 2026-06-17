import torch
import torch.nn as nn
import stanza
import os
import json

class POSUniversalWrapper(nn.Module):
    def __init__(self, model, input_names):
        super().__init__()
        self.model = model
        self.input_names = input_names
    
    def forward(self, *args):
        # Map args to names
        kwargs = {name: arg for name, arg in zip(self.input_names, args)}
        
        # Stanza POS forward: (word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens)
        # We need to provide them in order.
        
        # Helper to get arg or None
        def g(name): return kwargs.get(name)
        
        return self.model(
            g('word'), g('word_mask'), g('wordchars'), g('wordchars_mask'),
            g('upos'), g('xpos'), g('ufeats'), g('pretrained'),
            g('word_orig_idx'), g('sentlens'), g('wordlens'), "dummy"
        )

def export_pos_universal(lang):
    print(f"Exporting Universal POS for: {lang}")
    try:
        nlp = stanza.Pipeline(lang, processors='tokenize,pos', download_method=None)
        pos_model = nlp.processors['pos'].trainer.model
        pos_model.cpu()
        pos_model.eval()
        
        batch_size, seq_len = 1, 10
        
        # Basic inputs
        word = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
        word_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        pretrained = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
        word_orig_idx = torch.zeros((batch_size, seq_len), dtype=torch.long)
        sentlens = torch.tensor([seq_len], dtype=torch.long)
        wordlens = torch.ones((batch_size,), dtype=torch.long)
        
        input_names = ['word', 'word_mask', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens']
        inputs = [word, word_mask, pretrained, word_orig_idx, sentlens, wordlens]
        
        # Check for wordchars
        wordchars = None
        wordchars_mask = None
        if hasattr(pos_model, 'charmodel') and pos_model.charmodel:
            print(f"  {lang} has charmodel")
            wordchars = torch.randint(0, 100, (batch_size, seq_len, 10), dtype=torch.long)
            wordchars_mask = torch.ones((batch_size, seq_len, 10), dtype=torch.bool)
            input_names += ['wordchars', 'wordchars_mask']
            inputs += [wordchars, wordchars_mask]
            
        wrapper = POSUniversalWrapper(pos_model, input_names)
        
        output_dir = "./onnx_final"
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, f"{lang}_pos.onnx")
        
        torch.onnx.export(
            wrapper, tuple(inputs), onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=input_names, output_names=['upos_logits'],
            dynamic_axes={name: {1: 'seq_len'} for name in input_names if 'word' in name}
        )
        print(f"✅ Saved {lang}_pos.onnx")
    except Exception as e:
        print(f"❌ Failed {lang} POS: {e}")

langs = ['en', 'ja', 'zh-hans', 'vi', 'ar', 'es', 'ko']
for lang in langs:
    export_pos_universal(lang)
