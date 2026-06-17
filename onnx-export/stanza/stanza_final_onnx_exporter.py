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

# --- 2. Dynamic Wrappers ---

class ONNXCharacterModel(nn.Module):
    def __init__(self, original_char):
        super().__init__()
        self.char_emb = original_char.char_emb
        self.charlstm = original_char.charlstm.lstm # Use underlying nn.LSTM
        self.dropout = original_char.dropout
        self.char_attn = getattr(original_char, 'char_attn', None)

    def forward(self, chars):
        embs = self.dropout(self.char_emb(chars))
        res, _ = self.charlstm(embs)
        if self.char_attn:
            attn_weights = torch.sigmoid(self.char_attn(self.dropout(res)))
            res = res * attn_weights
        return res.sum(dim=1)

class POSONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.args = model.args
        self.word_emb = model.word_emb
        self.pretrained_emb = model.pretrained_emb
        self.trans_pretrained = model.trans_pretrained
        
        # ESSENTIAL: Use wrapped charmodel
        if model.args['char'] and hasattr(model, 'charmodel'):
            self.charmodel = ONNXCharacterModel(model.charmodel)
            self.trans_char = getattr(model, 'trans_char', None)
        else:
            self.charmodel = None
            
        self.taggerlstm = model.taggerlstm
        self.upos_hid = model.upos_hid
        self.upos_clf = model.upos_clf
        self.drop = model.drop

    def forward(self, word, wordchars, pretrained):
        inputs = []
        inputs.append(self.word_emb(word))
        inputs.append(self.trans_pretrained(self.pretrained_emb(pretrained)))
        
        if self.charmodel and self.trans_char:
            char_reps = self.charmodel(wordchars)
            # Make sure reshaping is dynamic
            char_reps = char_reps.reshape(word.size(0), word.size(1), -1)
            inputs.append(self.trans_char(self.drop(char_reps)))
            
        lstm_inputs = torch.cat(inputs, dim=-1)
        lstm_inputs = self.drop(lstm_inputs)
        
        lstm_outputs, _ = self.taggerlstm(lstm_inputs)
        hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        return self.upos_clf(self.drop(hid))

def export_pos(processor, lang, output_dir):
    print(f"Exporting Truly Complete POS model (V9) for {lang}...")
    model = processor.trainer.model.eval().cpu()
    
    for m in model.modules():
        if isinstance(m, nn.Embedding): m.weight.requires_grad = True
    
    wrapper = POSONNXWrapper(model)
    
    batch, seq, max_char = 1, 1, 5
    word = torch.randint(0, model.word_emb.num_embeddings, (batch, seq), dtype=torch.long)
    wordchars = torch.randint(0, 100, (batch * seq, max_char), dtype=torch.long)
    pretrained = torch.randint(0, model.pretrained_emb.num_embeddings, (batch, seq), dtype=torch.long)
    
    onnx_path = os.path.join(output_dir, f"{lang}_pos.onnx")
    torch.onnx.export(wrapper, (word, wordchars, pretrained), onnx_path,
                      opset_version=14, export_params=True, do_constant_folding=False,
                      input_names=['word', 'wordchars', 'pretrained'],
                      output_names=['upos_logits'],
                      dynamic_axes={
                          'word': {0: 'batch', 1: 'seq'}, 
                          'wordchars': {0: 'batch_seq'},
                          'pretrained': {0: 'batch', 1: 'seq'},
                          'upos_logits': {0: 'batch', 1: 'seq'}
                      })
    print(f"✅ V9 POS Success: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.2f} MB)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ko", help="Language code")
    parser.add_argument("--output_dir", default="./onnx_final", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"--- Truly Dynamic Stanza-to-ONNX Exporter V9 ---")
    nlp = stanza.Pipeline(args.lang, processors='tokenize,pos,lemma,depparse', download_method=None, device='cpu')
    if 'pos' in nlp.processors: export_pos(nlp.processors['pos'], args.lang, args.output_dir)

if __name__ == "__main__":
    main()
