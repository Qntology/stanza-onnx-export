import stanza
import os
import torch
import torch.nn as nn
import json
import traceback

# Monkey-patch pad_sequence for ONNX export if needed
original_pad_sequence = torch.nn.utils.rnn.pad_sequence
def patched_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if not torch.jit.is_tracing():
        return original_pad_sequence(sequences, batch_first, padding_value)
    
    # Simple implementation for tracing
    max_len = max(seq.size(0) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        pad_amount = max_len - seq.size(0)
        if pad_amount > 0:
            # Use constant padding for tracing
            padded = torch.nn.functional.pad(seq, (0, 0, 0, pad_amount) if seq.dim() > 1 else (0, pad_amount), value=padding_value)
            padded_seqs.append(padded)
        else:
            padded_seqs.append(seq)
    return torch.stack(padded_seqs, dim=0 if batch_first else 1)

torch.nn.utils.rnn.pad_sequence = patched_pad_sequence

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
        # Dummy text for charlm/bert
        text = [['dummy'] * seq_len for _ in range(batch_size)]
        
        # Determine if we need to pass text
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

def process_lang(lang):
    print(f"\n🚀 Processing: {lang}")
    try:
        # Load with default processors
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

        # Common inputs
        batch_size, seq_len = 1, 5
        dummy_word = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.long)
        dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        
        # Wordchars: (total_words, max_word_len)
        total_words = batch_size * seq_len
        max_word_len = 10
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

# Try important languages
for lang in ['ko', 'en', 'ja']:
    process_lang(lang)
