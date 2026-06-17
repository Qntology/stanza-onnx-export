import os
import requests
import zipfile
import glob
import torch
import torch.nn as nn
import shutil
import argparse
import logging
import traceback
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.common.vocab import CompositeVocab
from stanza.models.pos.trainer import Trainer as POSTrainer
from stanza.models.depparse.trainer import Trainer as DepparseTrainer
from stanza.models.lemma.trainer import Trainer as LemmaTrainer
from stanza.models.pos.vocab import MultiVocab

# --- Monkey Patching MultiVocab for compatibility ---
original_load_state_dict = MultiVocab.load_state_dict
def patched_load_state_dict(state_dict):
    if isinstance(state_dict, dict) and '_key2class' not in state_dict:
        state_dict['_key2class'] = {k: k for k in state_dict.keys() if k != '_key2class'}
    return original_load_state_dict(state_dict)
MultiVocab.load_state_dict = patched_load_state_dict

# --- Wrappers ---

class POSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens):
        return self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, None)

class DepparseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens):
        return self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, None)

# --- Logic to infer shapes from model layers ---

def infer_composite_dim(module):
    if isinstance(module, nn.ModuleList):
        return len(module)
    return None

def export_tokenizer(model, onnx_path):
    model.eval()
    emb_dim = model.embeddings.embedding_dim
    rnn_input_dim = model.rnn.input_size
    feat_dim = rnn_input_dim - emb_dim
    batch_size, seq_len = 1, 16
    dummy_input = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    dummy_feats = torch.zeros(batch_size, seq_len, feat_dim, dtype=torch.float)
    dummy_lengths = torch.tensor([seq_len], dtype=torch.long)
    torch.onnx.export(
        model, (dummy_input, dummy_feats, dummy_lengths), onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input', 'feats', 'lengths'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size', 1: 'seq'}, 'feats': {0: 'batch_size', 1: 'seq'}, 'lengths': {0: 'batch_size'}, 'output': {0: 'batch_size', 1: 'seq'}}
    )

def export_pos_robust(model, onnx_path):
    model.eval()
    wrapper = POSWrapper(model)
    batch_size, seq_len = 1, 8
    total_words = batch_size * seq_len
    max_char_len = 5
    
    word = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    word_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    wordchars = torch.randint(0, 10, (total_words, max_char_len), dtype=torch.long)
    wordchars_mask = torch.ones((total_words, max_char_len), dtype=torch.bool)
    upos = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    xpos_dim = infer_composite_dim(getattr(model, 'xpos_clf', None)) or infer_composite_dim(getattr(model, 'xpos_hid', None))
    xpos = torch.zeros((batch_size, seq_len, xpos_dim), dtype=torch.long) if xpos_dim else torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    feats_dim = infer_composite_dim(getattr(model, 'ufeats_clf', None))
    ufeats = torch.zeros((batch_size, seq_len, feats_dim), dtype=torch.long) if feats_dim else torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    pretrained = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    word_orig_idx = torch.arange(total_words, dtype=torch.long)
    sentlens = torch.tensor([seq_len], dtype=torch.long)
    wordlens = torch.ones((total_words,), dtype=torch.long) * max_char_len
    
    inputs = (word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens)
    torch.onnx.export(
        wrapper, inputs, onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens'],
        output_names=['loss', 'preds'],
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

def export_depparse_robust(model, onnx_path):
    model.eval()
    wrapper = DepparseWrapper(model)
    batch_size, seq_len = 1, 8
    total_words = batch_size * seq_len
    max_char_len = 5
    
    word = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    word_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    wordchars = torch.randint(0, 10, (total_words, max_char_len), dtype=torch.long)
    wordchars_mask = torch.ones((total_words, max_char_len), dtype=torch.bool)
    upos = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    xpos_dim = infer_composite_dim(getattr(model, 'xpos_emb', None))
    xpos = torch.zeros((batch_size, seq_len, xpos_dim), dtype=torch.long) if xpos_dim else torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    feats_dim = infer_composite_dim(getattr(model, 'ufeats_emb', None))
    ufeats = torch.zeros((batch_size, seq_len, feats_dim), dtype=torch.long) if feats_dim else torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    pretrained = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
    lemma = torch.zeros((batch_size, seq_len), dtype=torch.long)
    head = torch.zeros((batch_size, seq_len), dtype=torch.long)
    deprel = torch.zeros((batch_size, seq_len), dtype=torch.long)
    word_orig_idx = torch.arange(total_words, dtype=torch.long)
    sentlens = torch.tensor([seq_len], dtype=torch.long)
    wordlens = torch.ones((total_words,), dtype=torch.long) * max_char_len
    
    inputs = (word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)
    torch.onnx.export(
        wrapper, inputs, onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'lemma', 'head', 'deprel', 'word_orig_idx', 'sentlens', 'wordlens'],
        output_names=['loss', 'preds'],
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="ko", help="Language code")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    extracted_path = f"./temp_stanza_models/{args.lang}"
    pretrain_files = glob.glob(os.path.join(extracted_path, "pretrain/*.pt"))
    pretrain = Pretrain(filename=pretrain_files[0]) if pretrain_files else None
    foundation_cache = FoundationCache()
    output_dir = "./onnx_definitive_output"
    os.makedirs(output_dir, exist_ok=True)
    
    pt_files = glob.glob(os.path.join(extracted_path, "**/*.pt"), recursive=True)
    for pt_path in pt_files:
        if "pretrain" in pt_path: continue
        file_name = os.path.basename(pt_path)
        onnx_path = os.path.join(output_dir, f"{args.lang}_{file_name}.onnx")
        print(f"\n--- Processing {file_name} ---")
        
        # 1. Tokenizer
        try:
            trainer = TokenizerTrainer(model_file=pt_path, device='cpu')
            export_tokenizer(trainer.model, onnx_path)
            print(f"✅ Success: Tokenizer")
            continue
        except Exception: pass

        # 2. POS
        try:
            trainer = POSTrainer(model_file=pt_path, pretrain=pretrain, foundation_cache=foundation_cache, device='cpu')
            export_pos_robust(trainer.model, onnx_path)
            print(f"✅ Success: POS")
            continue
        except Exception as e:
            print(f"POS fail: {e}")

        # 3. Depparse
        try:
            trainer = DepparseTrainer.load(filename=pt_path, pretrain=pretrain, device='cpu', foundation_cache=foundation_cache)
            export_depparse_robust(trainer.model, onnx_path)
            print(f"✅ Success: Depparse")
            continue
        except Exception as e:
            print(f"Depparse fail: {e}")

if __name__ == "__main__":
    main()
