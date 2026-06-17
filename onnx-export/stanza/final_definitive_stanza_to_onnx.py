import stanza
import os
import torch
import torch.nn as nn
import json
import traceback
import inspect
import stanza.models.common.utils as stanza_utils

# --- 1. CRITICAL PATCHES FOR ONNX ---

def traceable_unsort(sorted_tensor, oidx):
    if not torch.jit.is_tracing():
        backidx = [x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]
        return sorted_tensor[backidx]
    backidx = torch.argsort(oidx)
    return sorted_tensor[backidx]

stanza_utils.unsort = traceable_unsort

# --- 2. WRAPPERS ---

class UniversalWrapper(nn.Module):
    def __init__(self, model, input_names):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.forward_params = list(inspect.signature(model.forward).parameters.keys())
    
    def forward(self, *args):
        kwargs = {name: arg for name, arg in zip(self.input_names, args)}
        call_args = []
        for p_name in self.forward_params:
            if p_name == 'self': continue
            if p_name == 'text':
                batch_size = args[0].size(0) if len(args) > 0 else 1
                seq_len = args[0].size(1) if len(args) > 0 else 1
                call_args.append([['dummy'] * seq_len for _ in range(batch_size)])
            elif p_name in kwargs:
                call_args.append(kwargs[p_name])
            else:
                call_args.append(None)
        outputs = self.model(*call_args)
        if isinstance(outputs, tuple): return tuple(outputs[1])
        return tuple(outputs)

class TokenizerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, feats, lengths):
        return self.model(x, feats, lengths)

# --- 3. EXPORT LOGIC ---

def process_lang(lang):
    print(f"\n🚀 Final Processing: {lang}")
    ONNX_DIR = "./onnx_final"
    VOCAB_DIR = "./vocabs"
    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(VOCAB_DIR, exist_ok=True)
    
    try:
        # STRATEGY: Force nocharlm to avoid pad_sequence errors in en/ja/etc
        processors_dict = {'tokenize': 'default', 'pos': 'default', 'lemma': 'default', 'depparse': 'default'}
        try:
            resources = stanza.resources.common.load_resources_json()
            lang_res = resources.get(lang, {})
            # Look for nocharlm for POS and Depparse
            for proc in ['pos', 'depparse']:
                proc_res = lang_res.get(proc, {})
                default_pkg = lang_res.get('default_processors', {}).get(proc, 'default')
                if f"{default_pkg}_nocharlm" in proc_res:
                    processors_dict[proc] = f"{default_pkg}_nocharlm"
                elif "combined_nocharlm" in proc_res:
                    processors_dict[proc] = "combined_nocharlm"
        except: pass

        print(f"  Using Processors: {processors_dict}")
        stanza.download(lang, processors=processors_dict)
        nlp = stanza.Pipeline(lang, processors=processors_dict)

        pos_model = nlp.processors['pos'].trainer.model
        dep_model = nlp.processors['depparse'].trainer.model
        
        # Vocab
        print(f"  Extracting Vocab...")
        vocab_data = {
            "upos": pos_model.vocab['upos']._id2unit,
            "deprel": dep_model.vocab['deprel']._id2unit,
            "word2id": pos_model.vocab['word']._unit2id,
        }
        with open(f"{VOCAB_DIR}/{lang}_vocab_full.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False)

        # Tokenizer
        print(f"  Exporting Tokenizer...")
        tok_model = nlp.processors['tokenize'].trainer.model
        embed_dim = tok_model.embeddings.weight.shape[1]
        feat_dim = tok_model.rnn.weight_ih_l0.shape[1] - embed_dim
        dummy_x = torch.randint(1, 10, (1, 10), dtype=torch.long)
        dummy_feats = torch.zeros((1, 10, feat_dim))
        dummy_lens = torch.tensor([10], dtype=torch.long)
        torch.onnx.export(
            TokenizerWrapper(tok_model.cpu().eval()), (dummy_x, dummy_feats, dummy_lens),
            f"{ONNX_DIR}/{lang}_tokenizer.onnx", opset_version=14, do_constant_folding=True,
            input_names=['x', 'feats', 'lengths'], output_names=['output'],
            dynamic_axes={'x': {1: 'seq_len'}, 'feats': {1: 'seq_len'}, 'output': {1: 'seq_len'}}
        )

        # Shared Inputs
        batch, seq = 1, 8
        total_w = batch * seq
        common_inputs = {
            'word': torch.randint(1, 10, (batch, seq), dtype=torch.long),
            'word_mask': torch.ones((batch, seq), dtype=torch.bool),
            'pretrained': torch.randint(1, 10, (batch, seq), dtype=torch.long),
            'word_orig_idx': torch.arange(total_w, dtype=torch.long),
            'sentlens': torch.tensor([seq] * batch, dtype=torch.long),
            'wordlens': torch.full((total_w,), 5, dtype=torch.long)
        }
        
        # POS
        print(f"  Exporting POS...")
        pos_names = list(common_inputs.keys())
        pos_vals = [common_inputs[n] for n in pos_names]
        if hasattr(pos_model, 'charmodel') and pos_model.charmodel:
            pos_names += ['wordchars', 'wordchars_mask']
            pos_vals += [torch.randint(1, 10, (total_w, 5), dtype=torch.long), torch.ones((total_w, 5), dtype=torch.bool)]
        
        torch.onnx.export(
            UniversalWrapper(pos_model.cpu().eval(), pos_names), tuple(pos_vals),
            f"{ONNX_DIR}/{lang}_pos.onnx", opset_version=14, do_constant_folding=True,
            input_names=pos_names, output_names=['upos_preds', 'xpos_preds', 'ufeats_preds'],
            dynamic_axes={'word': {1: 'seq_len'}, 'word_mask': {1: 'seq_len'}}
        )

        # Depparse
        print(f"  Exporting Depparse...")
        # Add root for depparse
        d_seq = seq + 1
        d_total = batch * d_seq
        dep_inputs = {
            'word': torch.randint(1, 10, (batch, d_seq), dtype=torch.long),
            'word_mask': torch.ones((batch, d_seq), dtype=torch.bool),
            'pretrained': torch.randint(1, 10, (batch, d_seq), dtype=torch.long),
            'word_orig_idx': torch.arange(d_total, dtype=torch.long),
            'sentlens': torch.tensor([d_seq] * batch, dtype=torch.long),
            'wordlens': torch.full((d_total,), 5, dtype=torch.long),
            'lemma': torch.randint(1, 10, (batch, d_seq), dtype=torch.long),
            'head': torch.zeros((batch, d_seq), dtype=torch.long),
            'deprel': torch.zeros((batch, d_seq), dtype=torch.long)
        }
        dep_names = list(dep_inputs.keys())
        dep_vals = [dep_inputs[n] for n in dep_names]
        if hasattr(dep_model, 'charmodel') and dep_model.charmodel:
            dep_names += ['wordchars', 'wordchars_mask']
            dep_vals += [torch.randint(1, 10, (d_total, 5), dtype=torch.long), torch.ones((d_total, 5), dtype=torch.bool)]

        torch.onnx.export(
            UniversalWrapper(dep_model.cpu().eval(), dep_names), tuple(dep_vals),
            f"{ONNX_DIR}/{lang}_depparse.onnx", opset_version=14, do_constant_folding=True,
            input_names=dep_names, output_names=['unlabeled_scores', 'deprel_scores'],
            dynamic_axes={'word': {1: 'seq_len'}, 'word_mask': {1: 'seq_len'}}
        )

        print(f"✅ Finished: {lang}")
        return True
    except Exception as e:
        print(f"❌ Failed: {lang} - {e}")
        return False

for lang in ['ko', 'en', 'ja']:
    process_lang(lang)
