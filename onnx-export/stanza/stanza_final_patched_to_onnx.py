import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import stanza
from stanza.models.common.char_model import CharacterModel
from stanza.models.pos.model import Tagger
from stanza.models.depparse.model import EmbeddingParser, GraphParser
from stanza.models.common.vocab import CompositeVocab
from stanza.models.common.packed_lstm import PackedLSTM
from stanza.models.common.hlstm import HighwayLSTM
import argparse

# --- Patches ---

def patched_packed_lstm_forward(self, input, lengths, hx=None):
    return self.lstm(input, hx)

PackedLSTM.forward = patched_packed_lstm_forward

def patched_hlstm_forward(self, input, sentlens, hx=None):
    curr_input = input
    for i in range(len(self.lstm)):
        output, _ = self.lstm[i](curr_input, sentlens)
        gate = torch.sigmoid(self.gate[i](curr_input))
        highway = torch.tanh(self.highway[i](curr_input))
        curr_input = output + gate * highway
        curr_input = self.drop(curr_input)
    return curr_input, None

HighwayLSTM.forward = patched_hlstm_forward

def patched_char_forward(self, chars, chars_mask, word_orig_idx, sentlens, wordlens):
    embs = self.dropout(self.char_emb(chars))
    res, _ = self.charlstm(embs, wordlens)
    if hasattr(res, 'data'): res = res.data
    if res.dim() == 2:
        res = res.view(chars.size(0), chars.size(1), -1)
    if self.args.get('char_attn', True):
        attn_weights = torch.sigmoid(self.char_attn(self.dropout(res)))
        res = res * attn_weights
    return res.sum(dim=1)

CharacterModel.forward = patched_char_forward

def patched_pos_forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text):
    inputs = []
    if self.args['word_emb_dim'] > 0:
        inputs.append(self.word_emb(word))
    if self.args['pretrain']:
        inputs.append(self.trans_pretrained(self.pretrained_emb(pretrained)))
    if self.args['char'] and self.args['char_emb_dim'] > 0:
        char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
        char_reps = char_reps.view(word.size(0), word.size(1), -1)
        inputs.append(self.trans_char(self.drop(char_reps)))
        
    lstm_inputs = torch.cat(inputs, dim=-1)
    lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
    lstm_inputs = self.drop(lstm_inputs)
    
    lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens)
    if hasattr(lstm_outputs, 'data'): lstm_outputs = lstm_outputs.data
    if lstm_outputs.dim() == 2:
        lstm_outputs = lstm_outputs.view(word.size(0), word.size(1), -1)
        
    upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
    upos_pred = self.upos_clf(self.drop(upos_hid))
    xpos_hid = F.relu(self.xpos_hid(self.drop(lstm_outputs)))
    ufeats_hid = F.relu(self.ufeats_hid(self.drop(lstm_outputs)))
    
    return upos_pred, xpos_hid, ufeats_hid

Tagger.forward = patched_pos_forward

def patched_parser_embed(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
    inputs = []
    if self.args['pretrain']:
        p_emb = self.trans_pretrained(self.pretrained_emb(pretrained))
        if self.args.get('pretrain_root', False):
            root = self.root_pretrained.view(1, 1, -1).expand(p_emb.size(0), -1, -1)
            p_emb = torch.cat([root, p_emb], dim=1)
        inputs.append(p_emb)
    if self.args['word_emb_dim'] > 0:
        inputs.append(self.word_emb(word))
        inputs.append(self.lemma_emb(lemma))
        
    if self.args.get('tag_emb_dim', 0) > 0:
        pos_emb = 0
        if self.args.get('use_upos', True):
            pos_emb = self.upos_emb(upos)
        if self.args.get('use_xpos', True):
            if isinstance(self.vocab['xpos'], CompositeVocab):
                for i in range(len(self.vocab['xpos'])):
                    pos_emb = pos_emb + self.xpos_emb[i](xpos[:, :, i])
            else:
                pos_emb = pos_emb + self.xpos_emb(xpos)
        if self.args.get('use_upos', True) or self.args.get('use_xpos', True):
            inputs.append(pos_emb)
        if self.args.get('use_ufeats', True):
            feats_emb = 0
            for i in range(len(self.vocab['feats'])):
                feats_emb = feats_emb + self.ufeats_emb[i](ufeats[:, :, i])
            inputs.append(feats_emb)

    if self.args['char'] and self.args['char_emb_dim'] > 0:
        char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
        char_reps = char_reps.view(word.size(0), word.size(1), -1)
        inputs.append(self.trans_char(self.drop(char_reps)))
        
    # Strictly check alignment before concatenation
    min_seq = min(x.size(1) for x in inputs)
    aligned_inputs = [x[:, :min_seq, :] for x in inputs]
            
    lstm_inputs = torch.cat(aligned_inputs, dim=-1)
    lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
    lstm_inputs = self.drop(lstm_inputs)
    
    lstm_outputs, _ = self.parserlstm(lstm_inputs, sentlens)
    if hasattr(lstm_outputs, 'data'): lstm_outputs = lstm_outputs.data
    return lstm_outputs

EmbeddingParser.embed = patched_parser_embed

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ko")
    args = parser.parse_args()
    
    print(f"--- Absolute Definitive ONNX Export for {args.lang} ---")
    nlp = stanza.Pipeline(args.lang, processors='tokenize,pos,lemma,depparse', download_method=None, device='cpu')
    output_dir = "./onnx_final_definitive"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. POS
    print("Exporting POS...")
    pos_model = nlp.processors['pos'].trainer.model.eval()
    batch, seq, max_char = 1, 8, 5
    device = 'cpu'
    
    word = torch.ones((batch, seq), dtype=torch.long, device=device)
    wordchars = torch.ones((batch * seq, max_char), dtype=torch.long, device=device)
    pretrained = torch.ones((batch, seq), dtype=torch.long, device=device)
    word_orig_idx = torch.arange(batch * seq, dtype=torch.long, device=device)
    sentlens = torch.tensor([seq], dtype=torch.long, device=device)
    wordlens = torch.ones((batch * seq,), dtype=torch.long, device=device) * max_char
    
    class POSWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, w, wc, p, oi, sl, wl):
            return self.m(w, None, wc, None, None, None, None, p, oi, sl, wl, None)
            
    inputs = (word, wordchars, pretrained, word_orig_idx, sentlens, wordlens)
    torch.onnx.export(POSWrapper(pos_model), inputs, os.path.join(output_dir, "pos.onnx"), opset_version=14, input_names=['word', 'wordchars', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens'], output_names=['upos_logits', 'xpos_hid', 'ufeats_hid'], dynamic_axes={'word': {0: 'batch', 1: 'seq'}})
    print("✅ POS Success")
    
    # 2. Depparse
    print("Exporting Depparse...")
    dep_model = nlp.processors['depparse'].trainer.model.eval()
    
    xpos_num = len(dep_model.vocab['xpos']) if isinstance(dep_model.vocab['xpos'], CompositeVocab) else 1
    feats_num = len(dep_model.vocab['feats']) if isinstance(dep_model.vocab['feats'], CompositeVocab) else 1
    
    seq = 9 # Root + 8 words
    word = torch.ones((batch, seq), dtype=torch.long, device=device)
    wordchars = torch.ones((seq, max_char), dtype=torch.long, device=device)
    upos = torch.ones((batch, seq), dtype=torch.long, device=device)
    xpos = torch.ones((batch, seq, xpos_num), dtype=torch.long, device=device)
    ufeats = torch.ones((batch, seq, feats_num), dtype=torch.long, device=device)
    pretrained = torch.ones((batch, seq), dtype=torch.long, device=device)
    lemma = torch.ones((batch, seq), dtype=torch.long, device=device)
    word_orig_idx = torch.arange(seq, dtype=torch.long, device=device)
    sentlens = torch.tensor([seq], dtype=torch.long, device=device)
    wordlens = torch.ones((seq,), dtype=torch.long, device=device) * max_char
    
    # head target must be seq - 1
    head_target = torch.ones((batch, seq - 1), dtype=torch.long, device=device)
    
    class DepWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, w, wc, u, x, uf, p, l, h, oi, sl, wl):
            unlabeled, deprel, _, _, _ = self.m.forward_scores(w, None, wc, None, u, x, uf, p, l, h, None, oi, sl, wl, None)
            return unlabeled, deprel
            
    inputs = (word, wordchars, upos, xpos, ufeats, pretrained, lemma, head_target, word_orig_idx, sentlens, wordlens)
    torch.onnx.export(DepWrapper(dep_model), inputs, os.path.join(output_dir, "depparse.onnx"), opset_version=14, input_names=['word', 'wordchars', 'upos', 'xpos', 'ufeats', 'pretrained', 'lemma', 'head', 'word_orig_idx', 'sentlens', 'wordlens'], output_names=['unlabeled_scores', 'deprel_scores'], dynamic_axes={'word': {0: 'batch', 1: 'seq'}})
    print("✅ Depparse Success")

if __name__ == "__main__":
    main()
