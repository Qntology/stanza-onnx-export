import torch
import torch.nn as nn
import stanza
import os
import json

class TokenizerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        # Stanza tokenizers vary, but many use a basic sequence labeling or CNN
        return self.model(input_ids)

def export_tokenizer(lang):
    print(f"Exporting Tokenizer for: {lang}")
    try:
        nlp = stanza.Pipeline(lang, processors='tokenize')
        tokenizer = nlp.processors['tokenize'].trainer.model
        tokenizer.cpu()
        tokenizer.eval()
        
        # Dummy input: Usually long tensor of chars or words
        dummy_input = torch.randint(0, 100, (1, 50), dtype=torch.long)
        
        output_dir = "./onnx_final"
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, f"{lang}_tokenizer.onnx")
        
        torch.onnx.export(
            tokenizer, dummy_input, onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {1: 'seq_len'}, 'output': {1: 'seq_len'}}
        )
        print(f"✅ Saved {lang}_tokenizer.onnx")
    except Exception as e:
        print(f"❌ Failed {lang} tokenizer: {e}")

langs = ['en', 'ja', 'zh-hans', 'vi', 'ar', 'es', 'ko']
for lang in langs:
    export_tokenizer(lang)
