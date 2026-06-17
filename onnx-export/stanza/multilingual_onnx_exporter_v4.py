import stanza
import os
import torch
import torch.nn as nn
import json
import traceback
import stanza.models.common.utils as stanza_utils

# 1. Patch Stanza's 'unsort' to be traceable
def traceable_unsort(sorted_tensor, oidx):
    # original: backidx = [x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]
    # traceable:
    backidx = torch.argsort(oidx)
    return sorted_tensor[backidx]

stanza_utils.unsort = traceable_unsort

# 2. Patch pad_sequence globally
original_pad_sequence = torch.nn.utils.rnn.pad_sequence
def patched_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if not torch.jit.is_tracing():
        return original_pad_sequence(sequences, batch_first, padding_value)
    
    # Traceable version using stack and pad
    max_len = 0
    for s in sequences:
        if s.size(0) > max_len: max_len = s.size(0)
        
    padded = []
    for s in sequences:
        p = torch.nn.functional.pad(s, (0, 0, 0, max_len - s.size(0)) if s.dim() > 1 else (0, max_len - s.size(0)), value=padding_value)
        padded.append(p)
    return torch.stack(padded, dim=0 if batch_first else 1)

torch.nn.utils.rnn.pad_sequence = patched_pad_sequence
torch.pad_sequence = patched_pad_sequence # sometimes accessed this way

ONNX_DIR = "./onnx_final"
VOCAB_DIR = "./vocabs"
os.makedirs(ONNX_DIR, exist_ok=True)
os.makedirs(VOCAB_DIR, exist_ok=True)

class POSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens):
        batch_size = word.size(0)
        seq_len = word.size(1)
        text = [['dummy'] * seq_len for _ in range(batch_size)]
        use_text = hasattr(self.model, 'charmodel_forward') or (hasattr(self.model, 'bert_model') and self.model.bert_model is not None)
        
        # Stanza models return (loss, preds) or just preds depending on training mode
        res = self.model(
            word, word_mask, wordchars, wordchars_mask,
            None, None, None, pretrained,
            word_orig_idx, sentlens, wordlens, text if use_text else None
        )
        if isinstance(res, tuple):
            loss, preds = res
            return tuple(preds)
        return tuple(res)

class DepparseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens):
        batch_size = word.size(0)
        seq_len = word.size(1)
        text = [['dummy'] * seq_len for _ in range(batch_size)]
        use_text = hasattr(self.model, 'charmodel_forward') or (hasattr(self.model, 'bert_model') and self.model.bert_model is not None)
        
        res = self.model(
            word, word_mask, wordchars, wordchars_mask,
            None, None, None, pretrained,
            lemma, head, deprel, word_orig_idx, sentlens, wordlens, text if use_text else None
        )
        if isinstance(res, tuple):
            loss, preds = res
            return tuple(preds)
        return tuple(res)

def process_lang(lang):
    print(f"\n🚀 Processing: {lang}")
    try:
        stanza.download(lang, processors='tokenize,pos,lemma,depparse')
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')

        pos_model = nlp.processors['pos'].trainer.model
        dep_model = nlp.processors['depparse'].trainer.model
        pos_model.cpu().eval()
        dep_model.cpu().eval()

        # Extract Vocab
        print(f"  Extracting Vocab...")
        vocab_data = {
            "pos_upos": pos_model.vocab['upos']._id2unit,
            "dep_deprel": dep_model.vocab['deprel']._id2unit if 'deprel' in dep_model.vocab else [],
        }
        with open(f"{VOCAB_DIR}/{lang}_vocab_full.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False)

        # Common inputs - Fixed sequence length for stable tracing
        batch_size, seq_len = 1, 10
        dummy_word = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.long)
        dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        total_words = batch_size * seq_len
        max_word_len = 5
        dummy_wordchars = torch.randint(1, 10, (total_words, max_word_len), dtype=torch.long)
        dummy_wordchars_mask = torch.ones((total_words, max_word_len), dtype=torch.bool)
        dummy_pretrained = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.long)
        dummy_orig_idx = torch.arange(total_words, dtype=torch.long)
        dummy_sentlens = torch.tensor([seq_len] * batch_size, dtype=torch.long)
        dummy_wordlens = torch.full((total_words,), max_word_len, dtype=torch.long)
        dummy_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long)

        # Export POS
        print(f"  Exporting POS...")
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

        # Export Depparse
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
        # traceback.print_exc()
        return False

# Start with ko
process_lang('ko')
process_lang('en')
