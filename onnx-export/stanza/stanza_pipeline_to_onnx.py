import os
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import stanza
import numpy as np
import glob
import argparse
import logging

# --- Advanced RNN Bypass for ONNX ---

class DummyPackedSequence:
    def __init__(self, data, batch_sizes):
        self._data = data
        self.batch_sizes = batch_sizes
        self.sorted_indices = None
        self.unsorted_indices = None
    
    @property
    def data(self):
        # Stanza's layers expect 2D data from PackedSequence
        # (batch * seq, hidden)
        return self._data.reshape(-1, self._data.size(-1))
    
    def to(self, *args, **kwargs):
        return self

def patched_pack(input, lengths, batch_first=False, enforce_sorted=True):
    # For batch_size=1, packing is just a metadata wrapper
    return DummyPackedSequence(input, torch.tensor([input.size(1) if batch_first else input.size(0)]))

def patched_unpack(sequence, batch_first=False, padding_value=0.0, total_length=None):
    if isinstance(sequence, DummyPackedSequence):
        # Unpack back to 3D
        return sequence._data, sequence.batch_sizes
    return sequence, None

# Apply patches
rnn_utils.pack_padded_sequence = patched_pack
rnn_utils.pad_packed_sequence = patched_unpack

# --- Wrappers ---

class POSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.xpos_dim = 0
        if hasattr(model, 'xpos_clf') and isinstance(model.xpos_clf, nn.ModuleList):
            self.xpos_dim = len(model.xpos_clf)
        elif hasattr(model, 'xpos_hid') and isinstance(model.xpos_hid, nn.ModuleList):
            self.xpos_dim = len(model.xpos_hid)
            
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens):
        batch, seq = word.size(0), word.size(1)
        upos_target = torch.zeros((batch, seq), dtype=torch.long, device=word.device)
        xpos_target = torch.zeros((batch, seq, self.xpos_dim), dtype=torch.long, device=word.device) if self.xpos_dim > 0 else torch.zeros((batch, seq), dtype=torch.long, device=word.device)
        ufeats_target = torch.zeros((batch, seq, 1), dtype=torch.long, device=word.device)
        
        _, out = self.model(
            word, word_mask, wordchars, wordchars_mask, upos_target, xpos_target, ufeats_target, 
            pretrained, word_orig_idx, sentlens, wordlens, None
        )
        upos_logits, xpos_logits, feats_logits = out
        return upos_logits

class DepparseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.xpos_dim = len(model.xpos_emb) if hasattr(model, 'xpos_emb') and isinstance(model.xpos_emb, nn.ModuleList) else 0
        self.feats_dim = len(model.ufeats_emb) if hasattr(model, 'ufeats_emb') and isinstance(model.ufeats_emb, nn.ModuleList) else 0

    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, lemma, word_orig_idx, sentlens, wordlens):
        batch, seq = word.size(0), word.size(1)
        upos_target = torch.zeros((batch, seq), dtype=torch.long, device=word.device)
        xpos_target = torch.zeros((batch, seq, self.xpos_dim), dtype=torch.long, device=word.device) if self.xpos_dim > 0 else torch.zeros((batch, seq), dtype=torch.long, device=word.device)
        ufeats_target = torch.zeros((batch, seq, self.feats_dim), dtype=torch.long, device=word.device) if self.feats_dim > 0 else torch.zeros((batch, seq), dtype=torch.long, device=word.device)
        head_target = torch.zeros((batch, seq), dtype=torch.long, device=word.device)
        deprel_target = torch.zeros((batch, seq), dtype=torch.long, device=word.device)
        
        _, scores = self.model(
            word, word_mask, wordchars, wordchars_mask, upos_target, xpos_target, ufeats_target, 
            pretrained, lemma, head_target, deprel_target, word_orig_idx, sentlens, wordlens, None
        )
        unlabeled, _, _, _, _ = scores
        return unlabeled

# --- Export Logic ---

def export_pos(processor, lang, output_dir):
    model = processor.trainer.model.eval().cpu()
    wrapper = POSWrapper(model)
    batch, seq, max_char = 1, 2, 5
    total_words = batch * seq
    word = torch.ones((batch, seq), dtype=torch.long)
    word_mask = torch.zeros((batch, seq), dtype=torch.bool)
    wordchars = torch.ones((total_words, max_char), dtype=torch.long)
    wordchars_mask = torch.zeros((total_words, max_char), dtype=torch.bool)
    pretrained = torch.ones((batch, seq), dtype=torch.long)
    word_orig_idx = torch.arange(total_words, dtype=torch.long)
    sentlens = torch.tensor([seq], dtype=torch.long)
    wordlens = torch.ones((total_words,), dtype=torch.long) * max_char
    onnx_path = os.path.join(output_dir, f"stanza_{lang}_pos.onnx")
    inputs = (word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens)
    print(f"Exporting POS with Stealth-Shim...")
    torch.onnx.export(wrapper, inputs, onnx_path, export_params=True, opset_version=14, input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens'], output_names=['upos_logits'], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, dynamic_axes={'word': {0: 'batch', 1: 'seq'}})
    print(f"✅ POS Exported")

def export_depparse(processor, lang, output_dir):
    model = processor.trainer.model.eval().cpu()
    wrapper = DepparseWrapper(model)
    batch, seq, max_char = 1, 2, 5
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
    onnx_path = os.path.join(output_dir, f"stanza_{lang}_depparse.onnx")
    inputs = (word, word_mask, wordchars, wordchars_mask, pretrained, lemma, word_orig_idx, sentlens, wordlens)
    print(f"Exporting Depparse with Stealth-Shim...")
    torch.onnx.export(wrapper, inputs, onnx_path, export_params=True, opset_version=14, input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'pretrained', 'lemma', 'word_orig_idx', 'sentlens', 'wordlens'], output_names=['unlabeled_scores'], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, dynamic_axes={'word': {0: 'batch', 1: 'seq'}})
    print(f"✅ Depparse Exported")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ko", help="Language code")
    args = parser.parse_args()
    output_dir = "./onnx_stealth_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Loading Stanza Pipeline for {args.lang} ---")
    nlp = stanza.Pipeline(args.lang, processors='tokenize,pos,lemma,depparse', download_method=None)
    if 'pos' in nlp.processors:
        try: export_pos(nlp.processors['pos'], args.lang, output_dir)
        except Exception as e: print(f"❌ POS fail: {e}")
    if 'depparse' in nlp.processors:
        try: export_depparse(nlp.processors['depparse'], args.lang, output_dir)
        except Exception as e: print(f"❌ Depparse fail: {e}")

if __name__ == "__main__":
    main()
