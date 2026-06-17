import torch
import torch.nn as nn
import stanza
import os
import json
import stanza.models.common.utils as stanza_utils

# Patch Stanza's 'unsort' to be traceable
def traceable_unsort(sorted_tensor, oidx):
    backidx = torch.argsort(oidx)
    return sorted_tensor[backidx]
stanza_utils.unsort = traceable_unsort

class POSWrapper(nn.Module):
    def __init__(self, model, input_names):
        super().__init__()
        self.model = model
        self.input_names = input_names
    def forward(self, *args):
        kwargs = {name: arg for name, arg in zip(self.input_names, args)}
        def g(name): return kwargs.get(name)
        # Stanza POS signature
        res = self.model(
            g('word'), g('word_mask'), g('wordchars'), g('wordchars_mask'),
            None, None, None, g('pretrained'),
            g('word_orig_idx'), g('sentlens'), g('wordlens'), "dummy"
        )
        # res can be (loss, preds)
        if isinstance(res, tuple): return tuple(res[1])
        return tuple(res)

class DepparseWrapper(nn.Module):
    def __init__(self, model, input_names):
        super().__init__()
        self.model = model
        self.input_names = input_names
    def forward(self, *args):
        kwargs = {name: arg for name, arg in zip(self.input_names, args)}
        def g(name): return kwargs.get(name)
        # Stanza Depparse signature
        res = self.model(
            g('word'), g('word_mask'), g('wordchars'), g('wordchars_mask'),
            None, None, None, g('pretrained'),
            g('lemma'), g('head'), g('deprel'),
            g('word_orig_idx'), g('sentlens'), g('wordlens'), "dummy"
        )
        if isinstance(res, tuple): return tuple(res[1])
        return tuple(res)

def export_lang(lang):
    print(f"\n🚀 Processing: {lang}")
    ONNX_DIR = "./onnx_final"
    VOCAB_DIR = "./vocabs"
    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(VOCAB_DIR, exist_ok=True)
    
    try:
        stanza.download(lang, processors='tokenize,pos,lemma,depparse')
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
        
        pos_model = nlp.processors['pos'].trainer.model
        dep_model = nlp.processors['depparse'].trainer.model
        pos_model.cpu().eval()
        dep_model.cpu().eval()

        # Extract Vocab for Rust
        print(f"  Extracting Vocab...")
        vocab_data = {
            "upos": pos_model.vocab['upos']._id2unit,
            "deprel": dep_model.vocab['deprel']._id2unit,
        }
        with open(f"{VOCAB_DIR}/{lang}_vocab_full.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False)

        # Basic inputs
        batch_size, seq_len = 1, 10
        word = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.long)
        word_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        pretrained = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.long)
        word_orig_idx = torch.arange(seq_len, dtype=torch.long).repeat(batch_size, 1).view(-1)
        sentlens = torch.tensor([seq_len] * batch_size, dtype=torch.long)
        wordlens = torch.full((batch_size * seq_len,), 5, dtype=torch.long)
        dummy_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long)

        pos_input_names = ['word', 'word_mask', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens']
        pos_inputs = [word, word_mask, pretrained, word_orig_idx, sentlens, wordlens]
        
        if hasattr(pos_model, 'charmodel') and pos_model.charmodel:
            pos_input_names += ['wordchars', 'wordchars_mask']
            pos_inputs += [torch.randint(1, 10, (batch_size * seq_len, 5), dtype=torch.long), torch.ones((batch_size * seq_len, 5), dtype=torch.bool)]

        # POS Export
        print(f"  Exporting POS...")
        pos_wrapper = POSWrapper(pos_model, pos_input_names)
        torch.onnx.export(
            pos_wrapper, tuple(pos_inputs), f"{ONNX_DIR}/{lang}_pos.onnx",
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=pos_input_names, output_names=['upos_preds', 'xpos_preds', 'ufeats_preds'],
            dynamic_axes={'word': {1: 'seq_len'}, 'word_mask': {1: 'seq_len'}}
        )

        # Depparse Export
        print(f"  Exporting Depparse...")
        dep_input_names = pos_input_names + ['lemma', 'head', 'deprel']
        dep_inputs = pos_inputs + [dummy_tensor, dummy_tensor, dummy_tensor]
        dep_wrapper = DepparseWrapper(dep_model, dep_input_names)
        torch.onnx.export(
            dep_wrapper, tuple(dep_inputs), f"{ONNX_DIR}/{lang}_depparse.onnx",
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=dep_input_names, output_names=['unlabeled_scores', 'deprel_scores'],
            dynamic_axes={'word': {1: 'seq_len'}, 'word_mask': {1: 'seq_len'}}
        )

        print(f"✅ Finished: {lang}")
    except Exception as e:
        print(f"❌ Failed: {lang} - {e}")
        traceback.print_exc()

for lang in ['ko', 'en', 'ja']:
    export_lang(lang)
