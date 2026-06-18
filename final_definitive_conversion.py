import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.nn.utils.rnn import PackedSequence

# Extreme limits
sys.setrecursionlimit(100000)

# --- 1. PATCHES ---

class ONNXBilinear(nn.Module):
    def __init__(self, orig):
        super().__init__(); self.weight, self.bias = orig.weight, orig.bias
    def forward(self, x1, x2):
        if x1.dim() == 3: return torch.einsum('nli,oij,nlj->nlo', x1, self.weight, x2) + self.bias
        return torch.einsum('ni,oij,nj->no', x1, self.weight, x2) + self.bias

def patch_model_recursively(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Bilinear): setattr(module, name, ONNXBilinear(child))
        else: patch_model_recursively(child)

def apply_global_patches():
    import torch.nn.utils.rnn as rnn_utils
    rnn_utils.pack_padded_sequence = lambda input, lengths, batch_first=False, enforce_sorted=True: input if torch.jit.is_tracing() else rnn_utils.pack_padded_sequence(input, lengths, batch_first, enforce_sorted)
    rnn_utils.pad_packed_sequence = lambda sequence, batch_first=False, padding_value=0.0, total_length=None: (sequence, torch.tensor([sequence.size(1 if batch_first else 0)] * sequence.size(0 if batch_first else 1))) if torch.jit.is_tracing() else rnn_utils.pad_packed_sequence(sequence, batch_first, padding_value, total_length)
    import stanza.models.common.hlstm as hlstm
    import stanza.models.common.packed_lstm as packed_lstm
    def hlstm_forward(self, input, seqlens, hx=None):
        hs, cs = [], []
        for l in range(self.num_layers):
            if self.dropout > 0 and l > 0: input = self.drop(input)
            layer_hx = (hx[0][l * self.num_directions:(l+1)*self.num_directions], hx[1][l * self.num_directions:(l+1)*self.num_directions]) if hx is not None else None
            h, (ht, ct) = self.lstm[l].lstm(input, layer_hx)
            hs.append(ht); cs.append(ct)
            input = h + torch.sigmoid(self.gate[l](input)) * (self.highway[l](input) if self.highway_func is None else self.highway_func(self.highway[l](input)))
        return input, (torch.cat(hs, 0), torch.cat(cs, 0))
    hlstm.HighwayLSTM.forward = hlstm_forward
    packed_lstm.PackedLSTM.forward = lambda self, input, lengths, hx=None: self.lstm(input, hx)

# --- 2. EXPORT ---

import stanza

def convert_lang(lang):
    ONNX_DIR = "./onnx_final"
    print(f"\n🚀 {lang}")
    try: nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse', download_method=None)
    except: return

    # A. TOKENIZER
    if 'tokenize' in nlp.processors:
        try:
            print("  📦 Tokenizer...")
            m = nlp.processors['tokenize'].trainer.model.cpu().eval()
            in_size = m.rnn.input_size
            num_feats = in_size - 32
            inputs = (torch.zeros(1, 100).long(), torch.zeros(1, 100, num_feats).float(), torch.tensor([100]).long())
            torch.onnx.export(m, inputs, f"{ONNX_DIR}/{lang}_tokenizer.onnx", opset_version=14, 
                             input_names=['x','f','l'], output_names=['output'])
            print("    ✅ Done")
        except Exception as e: print(f"    ❌ {e}")

    # B. POS
    if 'pos' in nlp.processors:
        try:
            print("  📦 POS...")
            m = nlp.processors['pos'].trainer.model.cpu().eval()
            patch_model_recursively(m)
            class POSWrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, word, mask, chars, chars_mask, pre, oidx, slen, wlen):
                    ins = []
                    if self.m.args['word_emb_dim'] > 0: ins.append(self.m.word_emb(word))
                    if self.m.args['pretrain']: ins.append(self.m.trans_pretrained(self.m.pretrained_emb(pre)))
                    if hasattr(self.m, 'charmodel_forward'):
                        ins.append(torch.zeros(word.size(0), word.size(1), self.m.charmodel_forward.hidden_dim()))
                        ins.append(torch.zeros(word.size(0), word.size(1), self.m.charmodel_backward.hidden_dim()))
                    elif hasattr(self.m, 'charmodel'):
                        ins.append(torch.zeros(word.size(0), word.size(1), self.m.args['transformed_dim']))
                    lstm_out, _ = self.m.taggerlstm(torch.cat(ins, 2), slen, hx=(self.m.taggerlstm_h_init.expand(-1, word.size(0), -1).contiguous(), self.m.taggerlstm_c_init.expand(-1, word.size(0), -1).contiguous()))
                    return self.m.upos_clf(F.relu(self.m.upos_hid(lstm_out)))
            
            inputs = (torch.randint(1, 10, (1, 8)), torch.ones((1, 8), dtype=torch.bool), 
                      torch.zeros(8, 32).long(), torch.ones(8, 32).bool(), 
                      torch.randint(1, 10, (1, 8)), torch.arange(8), torch.tensor([8]), torch.full((8,), 5))
            
            torch.onnx.export(POSWrapper(m), inputs, f"{ONNX_DIR}/{lang}_pos.onnx", opset_version=14,
                             input_names=['word','mask','chars','chars_mask','pre','oidx','slen','wlen'],
                             output_names=['upos'])
            print("    ✅ Done")
        except Exception as e: print(f"    ❌ {e}")

    # C. DEPPARSE
    if 'depparse' in nlp.processors:
        try:
            print("  📦 Depparse...")
            m = nlp.processors['depparse'].trainer.model.cpu().eval()
            patch_model_recursively(m)
            class DepWrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, word, mask, upos, pre, lemma, slen):
                    ins = []
                    if self.m.args['pretrain']: ins.append(self.m.trans_pretrained(self.m.pretrained_emb(pre)))
                    if self.m.args['word_emb_dim'] > 0:
                        ins.append(self.m.word_emb(word))
                        ins.append(self.m.lemma_emb(lemma))
                    if self.m.args['tag_emb_dim'] > 0:
                        p_emb = self.m.upos_emb(upos)
                        if hasattr(self.m, 'xpos_emb'):
                            if isinstance(self.m.xpos_emb, nn.ModuleList):
                                for l in self.m.xpos_emb: p_emb = p_emb + l(upos)
                            else: p_emb = p_emb + self.m.xpos_emb(upos)
                        ins.append(p_emb)
                    if hasattr(self.m, 'charmodel_forward'):
                        ins.append(torch.zeros(word.size(0), word.size(1), self.m.charmodel_forward.hidden_dim() + self.m.charmodel_backward.hidden_dim()))
                    elif hasattr(self.m, 'charmodel'):
                        ins.append(torch.zeros(word.size(0), word.size(1), self.m.args['transformed_dim']))
                    
                    curr = torch.cat(ins, 2)
                    if curr.size(-1) < self.m.parserlstm.input_size:
                        curr = torch.cat([curr, torch.zeros(curr.size(0), curr.size(1), self.m.parserlstm.input_size - curr.size(-1))], 2)
                    
                    lstm_out, _ = self.m.parserlstm(curr, slen, hx=(self.m.parserlstm_h_init.expand(-1, word.size(0), -1).contiguous(), self.m.parserlstm_c_init.expand(-1, word.size(0), -1).contiguous()))
                    if hasattr(self.m, 'unlabeled'): 
                        return self.m.unlabeled(lstm_out, lstm_out).squeeze(3), self.m.deprel(lstm_out, lstm_out)
                    arc = self.m.arc_embedding(lstm_out, lstm_out)
                    return self.m.unlabeled_linear(arc).squeeze(3), self.m.deprel_linear(arc)
            
            inputs = (torch.randint(1, 10, (1, 9)), torch.ones((1, 9), dtype=torch.bool), 
                      torch.zeros(1, 9).long(), torch.randint(1, 10, (1, 9)), 
                      torch.randint(1, 10, (1, 9)), torch.tensor([9]))
            
            torch.onnx.export(DepWrapper(m), inputs, f"{ONNX_DIR}/{lang}_depparse.onnx", opset_version=14,
                             input_names=['word','mask','upos','pre','lemma','slen'],
                             output_names=['unlabeled','deprel'])
            print("    ✅ Done")
        except Exception as e: print(f"    ❌ {e}")

if __name__ == "__main__":
    apply_global_patches()
    for l in ['en', 'de', 'es', 'fr', 'ja', 'pt', 'ar', 'cs', 'it', 'ko', 'nl', 'zh-hans']: convert_lang(l)
