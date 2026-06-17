import stanza
import os
import torch
import torch.nn as nn
import json
import traceback

# Output directories
ONNX_DIR = "./onnx_final"
VOCAB_DIR = "./vocabs"
os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(VOCAB_DIR, exist_ok=True)

class TokenizerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, feats, lengths):
        return self.model(x, feats, lengths)

class POSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens):
        batch_size = word.size(0)
        seq_len = word.size(1)
        text = [['dummy'] * seq_len for _ in range(batch_size)]
        
        # We pass None for text if model doesn't use charlm/bert to avoid tracing issues
        # But wait, Tagger.forward uses text if self.args.get('charlm') or self.bert_model
        use_text = hasattr(self.model, 'charmodel_forward') or (hasattr(self.model, 'bert_model') and self.model.bert_model is not None)
        
        loss, preds = self.model(
            word, word_mask, wordchars, wordchars_mask,
            None, None, None, pretrained,
            word_orig_idx, sentlens, wordlens, text if use_text else None
        )
        return tuple(preds)

class DepparseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens):
        batch_size = word.size(0)
        seq_len = word.size(1)
        text = [['dummy'] * seq_len for _ in range(batch_size)]
        use_text = hasattr(self.model, 'charmodel_forward') or (hasattr(self.model, 'bert_model') and self.model.bert_model is not None)
        
        loss, preds = self.model(
            word, word_mask, wordchars, wordchars_mask,
            None, None, None, pretrained,
            lemma, head, deprel, word_orig_idx, sentlens, wordlens, text if use_text else None
        )
        return tuple(preds)

def process_lang(lang, use_nocharlm=True):
    print(f"\n🚀 Processing: {lang} (nocharlm={use_nocharlm})")
    try:
        # 1. Download & Load
        print(f"  Downloading/Loading Stanza model...")
        
        # Try to force nocharlm packages
        processors_dict = {'tokenize': 'default', 'pos': 'default', 'lemma': 'default', 'depparse': 'default'}
        if use_nocharlm:
            try:
                resources = stanza.resources.common.load_resources_json()
                lang_resources = resources.get(lang, {})
                for proc in ['pos', 'lemma', 'depparse']:
                    proc_resources = lang_resources.get(proc, {})
                    default_pkg = lang_resources.get('default_processors', {}).get(proc, 'default')
                    if f"{default_pkg}_nocharlm" in proc_resources:
                        processors_dict[proc] = f"{default_pkg}_nocharlm"
                    elif "combined_nocharlm" in proc_resources:
                        processors_dict[proc] = "combined_nocharlm"
            except: pass

        stanza.download(lang, processors=processors_dict)
        nlp = stanza.Pipeline(lang, processors=processors_dict)

        pos_model = nlp.processors['pos'].trainer.model
        dep_model = nlp.processors['depparse'].trainer.model
        
        pos_model.cpu().eval()
        dep_model.cpu().eval()

        # 2. Extract Vocabs
        print(f"  Extracting Vocab...")
        vocab_data = {
            "pos_upos": pos_model.vocab['upos']._id2unit,
            "dep_deprel": dep_model.vocab['deprel']._id2unit if 'deprel' in dep_model.vocab else [],
        }
        with open(f"{VOCAB_DIR}/{lang}_vocab_full.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False)

        # 3. Export POS
        print(f"  Exporting POS...")
        batch_size, seq_len = 1, 10
        dummy_word = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
        dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        dummy_wordchars = torch.randint(0, 10, (batch_size * seq_len, 10), dtype=torch.long)
        dummy_wordchars_mask = torch.ones((batch_size * seq_len, 10), dtype=torch.bool)
        dummy_pretrained = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long)
        dummy_orig_idx = torch.arange(batch_size * seq_len, dtype=torch.long)
        dummy_sentlens = torch.tensor([seq_len] * batch_size, dtype=torch.long)
        dummy_wordlens = torch.ones((batch_size * seq_len,), dtype=torch.long)
        dummy_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long)

        pos_wrapper = POSWrapper(pos_model)
        torch.onnx.export(
            pos_wrapper, 
            (dummy_word, dummy_mask, dummy_wordchars, dummy_wordchars_mask, dummy_pretrained, dummy_orig_idx, dummy_sentlens, dummy_wordlens),
            f"{ONNX_DIR}/{lang}_pos.onnx",
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens'],
            output_names=['upos_preds', 'xpos_preds', 'ufeats_preds'],
            dynamic_axes={'word': {1: 'seq_len'}, 'word_mask': {1: 'seq_len'}}
        )

        # 4. Export Depparse
        print(f"  Exporting Depparse...")
        dep_wrapper = DepparseWrapper(dep_model)
        torch.onnx.export(
            dep_wrapper,
            (dummy_word, dummy_mask, dummy_wordchars, dummy_wordchars_mask, dummy_pretrained, dummy_tensor, dummy_tensor, dummy_tensor, dummy_orig_idx, dummy_sentlens, dummy_wordlens),
            f"{ONNX_DIR}/{lang}_depparse.onnx",
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'pretrained', 'lemma', 'head', 'deprel', 'word_orig_idx', 'sentlens', 'wordlens'],
            output_names=['unlabeled_scores', 'deprel_scores'],
            dynamic_axes={'word': {1: 'seq_len'}, 'word_mask': {1: 'seq_len'}}
        )

        print(f"✅ Finished: {lang}")
        return True
    except Exception as e:
        print(f"❌ Failed: {lang} - {e}")
        return False

# Try ko, en, ja
for lang in ['ko', 'en', 'ja']:
    process_lang(lang)
