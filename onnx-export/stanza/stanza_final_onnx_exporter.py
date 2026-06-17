import sys
import os
import re

# --- 1. SETUP LOCAL STANZA SOURCE ---
STANZA_LOCAL_PATH = r"C:\Users\HP\Desktop\stanza\stanza-main"
if os.path.exists(STANZA_LOCAL_PATH):
    sys.path.insert(0, STANZA_LOCAL_PATH)
    print(f"✅ Using local Stanza source from: {STANZA_LOCAL_PATH}")

import stanza
import torch
import torch.nn as nn
import json
import traceback
import inspect
import importlib
import stanza.models.common.utils as stanza_utils

# --- 2. TRACEABLE PATCHES ---

def traceable_unsort(sorted_tensor, oidx):
    if not torch.jit.is_tracing():
        return sorted_tensor[[x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]]
    return sorted_tensor[torch.argsort(oidx)]
stanza_utils.unsort = traceable_unsort

def patched_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if not torch.jit.is_tracing():
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
    max_len = max(s.size(0) for s in sequences)
    padded = [torch.nn.functional.pad(s, (0, 0, 0, max_len - s.size(0)) if s.dim() > 1 else (0, max_len - s.size(0)), value=padding_value) for s in sequences]
    return torch.stack(padded, dim=0 if batch_first else 1)

torch.nn.utils.rnn.pad_sequence = patched_pad_sequence

# --- 3. DYNAMIC SKELETON WRAPPER ---

class DynamicModelWrapper(nn.Module):
    def __init__(self, model, input_names):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.sig = inspect.signature(model.forward)
        self.param_names = [p for p in self.sig.parameters.keys() if p != 'self']

    def forward(self, *args):
        input_map = {name: arg for name, arg in zip(self.input_names, args)}
        call_args = []
        for p_name in self.param_names:
            if p_name == 'text':
                batch_size = args[0].size(0) if len(args) > 0 else 1
                seq_len = args[0].size(1) if len(args) > 0 else 1
                call_args.append([['dummy'] * seq_len for _ in range(batch_size)])
            elif p_name in input_map:
                call_args.append(input_map[p_name])
            else:
                call_args.append(None)
        res = self.model(*call_args)
        if isinstance(res, tuple): return tuple(res[1])
        return tuple(res)

# --- 4. CODE-AWARE LOGIC ---

def analyze_source_for_quirks(model_class_name):
    """Reads Stanza source to find unusual forward arguments."""
    base_path = os.path.join(STANZA_LOCAL_PATH, "stanza", "models")
    search_map = {
        "Tagger": "pos/model.py",
        "GraphParser": "depparse/model.py"
    }
    if model_class_name in search_map:
        target_file = os.path.join(base_path, search_map[model_class_name])
        if os.path.exists(target_file):
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Find forward method
                match = re.search(r"def forward\(self, (.*?)\):", content)
                if match:
                    args = [a.strip().split("=")[0] for a in match.group(1).split(",")]
                    print(f"      Found source signature for {model_class_name}: {args}")
                    return args
    return None

# --- 5. EXPORT ENGINE ---

def convert_lang(lang):
    print(f"\n🌍 Processing: {lang}")
    ONNX_DIR = "./onnx_final"
    VOCAB_DIR = "./vocabs"
    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(VOCAB_DIR, exist_ok=True)
    
    try:
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
        
        # Vocab
        pos_m = nlp.processors['pos'].trainer.model
        dep_m = nlp.processors['depparse'].trainer.model
        vocab_data = {"upos": pos_m.vocab['upos']._id2unit, "deprel": dep_m.vocab['deprel']._id2unit, "word2id": pos_m.vocab['word']._unit2id}
        with open(f"{VOCAB_DIR}/{lang}_vocab_full.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False)

        def safe_export(model, name, is_dep=False):
            model.cpu().eval()
            batch, seq = 1, 8
            tw = batch * seq
            if is_dep: seq += 1; tw = batch * seq
            
            inputs = {'word': torch.randint(1, 10, (batch, seq), dtype=torch.long), 'word_mask': torch.ones((batch, seq), dtype=torch.bool), 'pretrained': torch.randint(1, 10, (batch, seq), dtype=torch.long), 'word_orig_idx': torch.arange(tw, dtype=torch.long), 'sentlens': torch.tensor([seq] * batch, dtype=torch.long), 'wordlens': torch.full((tw,), 5, dtype=torch.long)}
            if is_dep: inputs.update({'lemma': torch.randint(1, 10, (batch, seq), dtype=torch.long), 'head': torch.zeros((batch, seq), dtype=torch.long), 'deprel': torch.zeros((batch, seq), dtype=torch.long)})
            
            if hasattr(model, 'charmodel') and model.charmodel:
                inputs['wordchars'] = torch.randint(1, 10, (tw, 5), dtype=torch.long)
                inputs['wordchars_mask'] = torch.ones((tw, 5), dtype=torch.bool)
            
            try:
                # 1st attempt: Normal wrapper
                names = list(inputs.keys())
                torch.onnx.export(DynamicModelWrapper(model, names), tuple(inputs[n] for n in names), f"{ONNX_DIR}/{lang}_{name}.onnx", opset_version=14, do_constant_folding=True, input_names=names, output_names=['o1', 'o2', 'o3'] if not is_dep else ['o1', 'o2'], dynamic_axes={'word': {1: 'seq_len'}})
                return True
            except Exception as e:
                print(f"      ⚠️ {name} failed, applying code-aware strip...")
                # Analyze source to be sure
                analyze_source_for_quirks(type(model).__name__)
                
                # NUCLEAR: Physically strip CharLM and BERT
                if hasattr(model, 'charmodel'): model.charmodel = None
                if hasattr(model, 'bert_model'): model.bert_model = None
                if hasattr(model, 'args'): model.args['charlm'] = False; model.args['bert_model'] = None
                
                new_names = [n for n in inputs.keys() if 'wordchars' not in n]
                torch.onnx.export(DynamicModelWrapper(model, new_names), tuple(inputs[n] for n in new_names), f"{ONNX_DIR}/{lang}_{name}.onnx", opset_version=14, do_constant_folding=True, input_names=new_names, output_names=['o1', 'o2', 'o3'] if not is_dep else ['o1', 'o2'], dynamic_axes={'word': {1: 'seq_len'}})
                return True

        safe_export(pos_m, "pos")
        safe_export(dep_m, "depparse", is_dep=True)
        print(f"✅ Finished: {lang}")
        return True
    except Exception as e:
        print(f"❌ Failed: {lang} - {e}")
        return False

if __name__ == "__main__":
    langs = ['sq', 'ar', 'az', 'bn', 'bg', 'ca', 'zh-hans', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'ka', 'de', 'el', 'he', 'hi', 'hu', 'is', 'id', 'it', 'ja', 'kk', 'km', 'ko', 'lv', 'lt', 'ms', 'mr', 'no', 'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sk', 'sl', 'es', 'sw', 'sv', 'tl', 'te', 'th', 'tr', 'uk', 'ur', 'uz', 'vi']
    for lang in langs:
        if not os.path.exists(f"./onnx_final/{lang}_depparse.onnx"):
            convert_lang(lang)
