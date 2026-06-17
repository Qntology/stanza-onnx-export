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
        self.charlstm = original_char.charlstm.lstm
        self.dropout = original_char.dropout
        self.char_attn = getattr(original_char, 'char_attn', None)

    def forward(self, chars):
        embs = self.dropout(self.char_emb(chars))
        res, _ = self.charlstm(embs)
        if self.char_attn:
            attn_weights = torch.sigmoid(self.char_attn(self.dropout(res)))
            res = res * attn_weights
        return res.sum(dim=1)

class DepparseONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.args = model.args
        self.word_emb = model.word_emb
        self.lemma_emb = model.lemma_emb
        self.pretrained_emb = model.pretrained_emb
        self.trans_pretrained = model.trans_pretrained
        self.upos_emb = model.upos_emb
        self.xpos_emb = model.xpos_emb
        
        if model.args['char'] and hasattr(model, 'charmodel'):
            self.charmodel = ONNXCharacterModel(model.charmodel)
            self.trans_char = getattr(model, 'trans_char', None)
        else:
            self.charmodel = None
            
        self.parserlstm = model.parserlstm
        self.unlabeled = model.unlabeled
        self.deprel = model.deprel
        self.drop = model.drop

    def forward(self, word, wordchars, upos, xpos, pretrained, lemma):
        inputs = []
        inputs.append(self.trans_pretrained(self.pretrained_emb(pretrained)))
        inputs.append(self.word_emb(word))
        inputs.append(self.lemma_emb(lemma))
        
        pos_emb = self.upos_emb(upos)
        if isinstance(self.xpos_emb, nn.ModuleList):
            for i in range(len(self.xpos_emb)): pos_emb = pos_emb + self.xpos_emb[i](xpos[:, :, i])
        else: pos_emb = pos_emb + self.xpos_emb(xpos)
        
        inputs.append(pos_emb)
        if self.args.get('use_ufeats', True): inputs.append(pos_emb)
        
        if self.charmodel and self.trans_char:
            char_reps = self.charmodel(wordchars)
            char_reps = char_reps.reshape(word.size(0), word.size(1), -1)
            inputs.append(self.trans_char(self.drop(char_reps)))
            
        # Cat and alignment
        lstm_inputs = torch.cat(inputs, dim=-1)
        lstm_outputs, _ = self.parserlstm(lstm_inputs)
        
        unlabeled_scores = self.unlabeled(lstm_outputs, lstm_outputs)
        deprel_scores = self.deprel(lstm_outputs, lstm_outputs)
        return unlabeled_scores, deprel_scores

def export_depparse(processor, lang, output_dir):
    print(f"Exporting Fully Dynamic Depparse model (V10) for {lang}...")
    model = processor.trainer.model.eval().cpu()
    
    for m in model.modules():
        if isinstance(m, nn.Embedding): m.weight.requires_grad = True
    
    from stanza.models.common.biaffine import BiaffineScorer
    def patched_biaffine_forward(self, input1, input2):
        out = torch.matmul(input1, self.W)
        out = torch.matmul(out, input2.transpose(1, 2))
        if self.b is not None: out = out + self.b
        return out
    BiaffineScorer.forward = patched_biaffine_forward

    wrapper = DepparseONNXWrapper(model)
    # USE SEQ=2 TO FORCE DYNAMNIC RECOGNITION
    batch, seq, max_char = 1, 2, 5
    
    word = torch.randint(0, model.word_emb.num_embeddings, (batch, seq), dtype=torch.long)
    wordchars = torch.randint(0, 100, (batch * seq, max_char), dtype=torch.long)
    upos = torch.randint(0, model.upos_emb.num_embeddings, (batch, seq), dtype=torch.long)
    
    if isinstance(model.xpos_emb, nn.ModuleList):
        xpos = torch.cat([torch.randint(0, e.num_embeddings, (batch, seq, 1), dtype=torch.long) for e in model.xpos_emb], dim=-1)
    else:
        xpos = torch.randint(0, model.xpos_emb.num_embeddings, (batch, seq), dtype=torch.long)
        
    pretrained = torch.randint(0, model.pretrained_emb.num_embeddings, (batch, seq), dtype=torch.long)
    lemma = torch.randint(0, model.lemma_emb.num_embeddings, (batch, seq), dtype=torch.long)
    
    onnx_path = os.path.join(output_dir, f"{lang}_depparse.onnx")
    torch.onnx.export(wrapper, (word, wordchars, upos, xpos, pretrained, lemma), onnx_path,
                      opset_version=14, export_params=True, do_constant_folding=False,
                      input_names=['word', 'wordchars', 'upos', 'xpos', 'pretrained', 'lemma'],
                      output_names=['unlabeled_scores', 'deprel_scores'],
                      dynamic_axes={
                          'word': {0: 'batch', 1: 'seq'},
                          'wordchars': {0: 'batch_seq'},
                          'upos': {0: 'batch', 1: 'seq'},
                          'xpos': {0: 'batch', 1: 'seq'},
                          'pretrained': {0: 'batch', 1: 'seq'},
                          'lemma': {0: 'batch', 1: 'seq'},
                          'unlabeled_scores': {0: 'batch', 1: 'seq', 2: 'seq_heads'},
                          'deprel_scores': {0: 'batch', 1: 'seq', 2: 'seq_heads'}
                      })
    print(f"✅ V10 Depparse Success: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.2f} MB)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ko", help="Language code")
    parser.add_argument("--output_dir", default="./onnx_final", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    nlp = stanza.Pipeline(args.lang, processors='tokenize,pos,lemma,depparse', download_method=None, device='cpu')
    if 'depparse' in nlp.processors: export_depparse(nlp.processors['depparse'], args.lang, args.output_dir)

if __name__ == "__main__":
    main()
