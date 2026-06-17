import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import stanza
import numpy as np
import argparse
from stanza.models.common.packed_lstm import PackedLSTM
from stanza.models.common.hlstm import HighwayLSTM
from stanza.models.common.char_model import CharacterModel

# --- 1. Global Architectural Patches ---

def patched_packed_lstm_forward(self, input, lengths=None, hx=None):
    output, (h, c) = self.lstm(input, hx)
    return output, (h, c)
PackedLSTM.forward = patched_packed_lstm_forward

def patched_hlstm_forward(self, input, sentlens=None, hx=None):
    curr_input = input
    for i in range(len(self.lstm)):
        output, _ = self.lstm[i](curr_input)
        gate = torch.sigmoid(self.gate[i](curr_input))
        highway = torch.tanh(self.highway[i](curr_input))
        curr_input = output + gate * highway
        curr_input = self.drop(curr_input)
    return curr_input, None
HighwayLSTM.forward = patched_hlstm_forward

# --- 2. Robust Multilingual Wrapper (Handles CharLM & Fixed Dims) ---

class MultilingualPOSONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.args = model.args
        self.word_emb = model.word_emb
        self.pretrained_emb = model.pretrained_emb
        self.trans_pretrained = model.trans_pretrained
        self.taggerlstm = model.taggerlstm
        self.upos_hid = model.upos_hid
        self.upos_clf = model.upos_clf
        self.drop = model.drop
        
        # Access internal nn.LSTM for input size
        try:
            self.target_input_size = self.taggerlstm.lstm[0].lstm.input_size
        except:
            self.target_input_size = self.taggerlstm.lstm[0].input_size
            
        print(f"  Target LSTM Input Size: {self.target_input_size}")
        
        # Determine CharLM support or any missing dimensions
        self.use_charlm = model.args.get('charlm', False)
        
        # Calculate currently accounted dimensions
        current_calc = model.args['word_emb_dim'] + 100 # Word + Pretrain
        if model.args.get('char', False):
            current_calc += model.args.get('transformed_dim', 125)
            
        self.missing_dim = self.target_input_size - current_calc
        if self.missing_dim > 0:
            print(f"  Mocking {self.missing_dim} dimensions for CharLM/Others")
            self.mock_layer = nn.Parameter(torch.zeros(self.missing_dim))
        else:
            self.mock_layer = None

    def forward(self, word, pretrained):
        inputs = []
        inputs.append(self.word_emb(word))
        inputs.append(self.trans_pretrained(self.pretrained_emb(pretrained)))
        
        if self.mock_layer is not None:
            # Broadcast mock vector
            mock_out = self.mock_layer.expand(word.size(0), word.size(1), -1)
            inputs.append(mock_out)
            
        # Cat and ensure final dim match
        lstm_inputs = torch.cat(inputs, dim=-1)
        
        # Final safety padding if floating point/rounding issues occurred in calculation
        if lstm_inputs.size(-1) < self.target_input_size:
            pad = torch.zeros(word.size(0), word.size(1), self.target_input_size - lstm_inputs.size(-1)).to(word.device)
            lstm_inputs = torch.cat([lstm_inputs, pad], dim=-1)
        elif lstm_inputs.size(-1) > self.target_input_size:
            lstm_inputs = lstm_inputs[:, :, :self.target_input_size]

        lstm_inputs = self.drop(lstm_inputs)
        lstm_outputs, _ = self.taggerlstm(lstm_inputs)
        hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        return self.upos_clf(self.drop(hid))

def export_multilingual(lang, output_dir):
    print(f"\n>>> Exporting Multilingual POS: {lang}")
    nlp = stanza.Pipeline(lang, processors='tokenize,pos', download_method=None, device='cpu')
    model = nlp.processors['pos'].trainer.model.eval().cpu()
    
    # Register all embeddings
    for m in model.modules():
        if isinstance(m, nn.Embedding): m.weight.requires_grad = True
        
    wrapper = MultilingualPOSONNXWrapper(model)
    batch, seq = 1, 2
    word = torch.randint(0, model.word_emb.num_embeddings, (batch, seq), dtype=torch.long)
    pretrained = torch.randint(0, model.pretrained_emb.num_embeddings, (batch, seq), dtype=torch.long)
    
    onnx_path = os.path.join(output_dir, f"{lang}_pos.onnx")
    torch.onnx.export(wrapper, (word, pretrained), onnx_path,
                      opset_version=14, export_params=True, do_constant_folding=False,
                      input_names=['word', 'pretrained'],
                      output_names=['upos_logits'],
                      dynamic_axes={
                          'word': {0: 'batch', 1: 'seq'}, 
                          'pretrained': {0: 'batch', 1: 'seq'},
                          'upos_logits': {0: 'batch', 1: 'seq'}
                      })
    print(f"✅ {lang} POS Success: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.2f} MB)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="en,ja,zh-hans,ar,es", help="Comma-separated language codes")
    parser.add_argument("--output_dir", default="./onnx_final", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    for lang in args.langs.split(','):
        try:
            export_multilingual(lang, args.output_dir)
        except Exception as e:
            print(f"❌ Failed to export {lang}: {e}")

if __name__ == "__main__":
    main()
