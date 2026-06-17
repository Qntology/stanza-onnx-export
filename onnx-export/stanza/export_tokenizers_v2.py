import torch
import torch.nn as nn
import stanza
import os
import json

class TokenizerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, feats, lengths):
        return self.model(x, feats, lengths)

def export_tokenizer(lang):
    print(f"Exporting Tokenizer for: {lang}")
    try:
        nlp = stanza.Pipeline(lang, processors='tokenize', download_method=None)
        tokenizer = nlp.processors['tokenize'].trainer.model
        tokenizer.cpu()
        tokenizer.eval()
        
        # Detect dimensions
        embed_dim = tokenizer.embeddings.weight.shape[1]
        rnn_input_dim = tokenizer.rnn.weight_ih_l0.shape[1]
        feat_dim = rnn_input_dim - embed_dim
        print(f"  Embed: {embed_dim}, RNN Input: {rnn_input_dim}, Feat: {feat_dim}")
        
        wrapper = TokenizerWrapper(tokenizer)
        
        batch_size, seq_len = 1, 50
        x = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
        feats = torch.zeros((batch_size, seq_len, feat_dim), dtype=torch.long)
        lengths = torch.tensor([seq_len], dtype=torch.long)
        
        output_dir = "./onnx_final"
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, f"{lang}_tokenizer.onnx")
        
        torch.onnx.export(
            wrapper, (x, feats, lengths), onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['x', 'feats', 'lengths'], output_names=['output'],
            dynamic_axes={'x': {1: 'seq_len'}, 'feats': {1: 'seq_len'}, 'output': {1: 'seq_len'}}
        )
        print(f"✅ Saved {lang}_tokenizer.onnx")

        # Save config
        config = {
            "feat_dim": feat_dim,
            "embed_dim": embed_dim
        }
        config_path = os.path.join(output_dir, f"{lang}_tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        print(f"✅ Saved {lang}_tokenizer_config.json")
    except Exception as e:
        print(f"❌ Failed {lang} tokenizer: {e}")

langs = ['en', 'ja', 'zh-hans', 'vi', 'ar', 'es', 'ko']
for lang in langs:
    export_tokenizer(lang)
