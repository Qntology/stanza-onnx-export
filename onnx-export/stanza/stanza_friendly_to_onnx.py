import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import stanza
import numpy as np
import argparse

# --- ONNX-Friendly Overrides ---

class ONNXFriendlyTagger(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens):
        m = self.original_model
        inputs = []
        
        # 1. Word Embeddings
        if m.args['word_emb_dim'] > 0:
            inputs.append(m.word_emb(word))
            
        # 2. Pretrained Embeddings
        if m.args['pretrain']:
            p_emb = m.pretrained_emb(pretrained)
            inputs.append(m.trans_pretrained(p_emb))
            
        # 3. Char Embeddings
        if m.args['char'] and m.args['char_emb_dim'] > 0:
            # We verified no CharLM for Korean
            char_reps = m.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            # If charmodel returns PackedSequence, unpack it
            if hasattr(char_reps, 'data'):
                # We need to reshape back to (batch, seq, hidden)
                char_reps = char_reps.data.view(word.size(0), word.size(1), -1)
            inputs.append(m.trans_char(m.drop(char_reps)))
            
        # 4. LSTM
        lstm_inputs = torch.cat(inputs, dim=-1)
        lstm_inputs = m.drop(lstm_inputs)
        
        # HighwayLSTM handles raw tensors if we are careful
        # But for ONNX, we might need to bypass its packing
        # Let's call the parserlstm's internal nn.LSTM if possible
        # HighwayLSTM has self.taggerlstm which is a PackedLSTM or nn.LSTM
        # In Korean, it's nn.LSTM because rec_dropout=0
        
        # We'll use the original taggerlstm but pass raw tensor
        # HighwayLSTM.forward(input, sentlens, hx)
        lstm_outputs, _ = m.taggerlstm(lstm_inputs, sentlens)
        
        # If output is PackedSequence, unpack it
        if hasattr(lstm_outputs, 'data'):
            lstm_outputs = lstm_outputs.data.view(word.size(0), word.size(1), -1)
            
        # 5. Classifiers
        upos_hid = F.relu(m.upos_hid(m.drop(lstm_outputs)))
        upos_pred = m.upos_clf(m.drop(upos_hid))
        
        return upos_pred

class ONNXFriendlyParser(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, lemma, word_orig_idx, sentlens, wordlens):
        m = self.original_model
        inputs = []
        
        # 1. Embeddings
        if m.args['pretrain']:
            p_emb = m.trans_pretrained(m.pretrained_emb(pretrained))
            # Handle pretrain_root expansion
            if m.args.get('pretrain_root', False):
                # Expand root to match batch size
                root = m.root_pretrained.view(1, 1, -1).expand(word.size(0), -1, -1)
                p_emb = torch.cat([root, p_emb], dim=1)
                # If we pad pretrain, we must ensure other inputs are also padded or trimmed!
                # Actually, Stanza's parser pads others internally if needed? No.
                # Usually word/lemma already include the root.
            inputs.append(p_emb)
            
        if m.args['word_emb_dim'] > 0:
            inputs.append(m.word_emb(word))
            inputs.append(m.lemma_emb(lemma))
            
        # Concatenate everything
        # All elements in inputs must have same batch and seq dimensions!
        # If there's a mismatch (e.g. 10 vs 11), we pad the smaller ones.
        max_seq = max(x.size(1) for x in inputs)
        padded_inputs = []
        for x in inputs:
            if x.size(1) < max_seq:
                # Zero pad on sequence dimension
                padding = torch.zeros((x.size(0), max_seq - x.size(1), x.size(2)), device=x.device)
                padded_inputs.append(torch.cat([padding, x], dim=1)) # Pad at beginning (Root position)
            else:
                padded_inputs.append(x)
                
        lstm_inputs = torch.cat(padded_inputs, dim=-1)
        lstm_inputs = m.drop(lstm_inputs)
        
        # 2. LSTM
        lstm_outputs, _ = m.parserlstm(lstm_inputs, sentlens)
        if hasattr(lstm_outputs, 'data'):
            lstm_outputs = lstm_outputs.data.view(word.size(0), -1, lstm_outputs.size(-1))
            
        # 3. Scores
        if hasattr(m, 'arc_embedding'):
            arc_scores = m.arc_embedding(m.drop(lstm_outputs), m.drop(lstm_outputs))
            unlabeled_scores = m.unlabeled_linear(arc_scores).squeeze(3)
        else:
            unlabeled_scores = m.unlabeled(m.drop(lstm_outputs), m.drop(lstm_outputs)).squeeze(3)
            
        return unlabeled_scores

# --- Export ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ko", help="Language code")
    args = parser.parse_args()
    
    print(f"--- Loading Stanza Pipeline for {args.lang} ---")
    nlp = stanza.Pipeline(args.lang, processors='tokenize,pos,lemma,depparse', download_method=None)
    output_dir = "./onnx_friendly_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # POS
    print("Exporting Friendly POS...")
    pos_model = nlp.processors['pos'].trainer.model
    friendly_pos = ONNXFriendlyTagger(pos_model).eval().cpu()
    
    # Dummy inputs for Korean
    batch, seq, max_char = 1, 8, 5
    total_words = batch * seq
    word = torch.ones((batch, seq), dtype=torch.long)
    word_mask = torch.zeros((batch, seq), dtype=torch.bool)
    wordchars = torch.ones((total_words, max_char), dtype=torch.long)
    wordchars_mask = torch.zeros((total_words, max_char), dtype=torch.bool)
    pretrained = torch.ones((batch, seq), dtype=torch.long)
    word_orig_idx = torch.arange(total_words, dtype=torch.long)
    sentlens = torch.tensor([seq], dtype=torch.long)
    wordlens = torch.ones((total_words,), dtype=torch.long) * max_char
    
    inputs = (word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens)
    torch.onnx.export(friendly_pos, inputs, os.path.join(output_dir, "pos.onnx"), opset_version=14, input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens'], output_names=['upos_logits'], dynamic_axes={'word': {0: 'batch', 1: 'seq'}})
    print("✅ POS Exported")
    
    # Depparse
    print("Exporting Friendly Depparse...")
    dep_model = nlp.processors['depparse'].trainer.model
    friendly_dep = ONNXFriendlyParser(dep_model).eval().cpu()
    
    # Depparse often expects Root node at index 0
    seq = 9 # 8 words + 1 root
    total_words = seq
    word = torch.ones((batch, seq), dtype=torch.long)
    word_mask = torch.zeros((batch, seq), dtype=torch.bool)
    wordchars = torch.ones((total_words, max_char), dtype=torch.long)
    wordchars_mask = torch.zeros((total_words, max_char), dtype=torch.bool)
    pretrained = torch.ones((batch, seq), dtype=torch.long)
    lemma = torch.ones((batch, seq), dtype=torch.long)
    word_orig_idx = torch.arange(total_words, dtype=torch.long)
    sentlens = torch.tensor([seq], dtype=torch.long)
    wordlens = torch.ones((total_words,), dtype=torch.long) * max_char
    
    inputs = (word, word_mask, wordchars, wordchars_mask, pretrained, lemma, word_orig_idx, sentlens, wordlens)
    torch.onnx.export(friendly_dep, inputs, os.path.join(output_dir, "depparse.onnx"), opset_version=14, input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'pretrained', 'lemma', 'word_orig_idx', 'sentlens', 'wordlens'], output_names=['unlabeled_scores'], dynamic_axes={'word': {0: 'batch', 1: 'seq'}})
    print("✅ Depparse Exported")

if __name__ == "__main__":
    main()
