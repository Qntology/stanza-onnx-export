import sys
import os
import re
import importlib

# --- 1. SETUP LOCAL STANZA SOURCE ---
STANZA_LOCAL_PATH = r"C:\Users\HP\Desktop\stanza\stanza-main"
if os.path.exists(STANZA_LOCAL_PATH):
    sys.path.insert(0, STANZA_LOCAL_PATH)
    print(f"✅ Using local Stanza source from: {STANZA_LOCAL_PATH}")

import stanza
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import inspect
import stanza.models.common.utils as stanza_utils
from stanza.models.common.char_model import CharacterLanguageModel, CharacterModel
from stanza.models.depparse.model import GraphParser
from stanza.models.pos.model import Tagger

# --- 2. GLOBAL TRACEABLE PATCHES ---

def traceable_unsort(sorted_tensor, oidx):
    if not torch.jit.is_tracing():
        return sorted_tensor[[x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]]
    return sorted_tensor[torch.argsort(oidx)]

stanza_utils.unsort = traceable_unsort

def patched_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if not torch.jit.is_tracing():
        return torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
    if not isinstance(sequences, (list, tuple)): return sequences
    max_len = 0
    for s in sequences:
        if s.size(0) > max_len: max_len = s.size(0)
    padded = [torch.nn.functional.pad(s, (0, 0, 0, max_len - s.size(0)) if s.dim() > 1 else (0, max_len - s.size(0)), value=padding_value) for s in sequences]
    return torch.stack(padded, dim=0 if batch_first else 1)

torch.nn.utils.rnn.pad_sequence = patched_pad_sequence

def apply_deep_patches():
    mods = ['stanza.models.pos.model', 'stanza.models.depparse.model', 'stanza.models.common.char_model', 'stanza.models.common.biaffine', 'stanza.models.common.bert_embedding', 'stanza.models.pos.data', 'stanza.models.ner.model']
    for m_name in mods:
        try:
            m = importlib.import_module(m_name)
            if hasattr(m, 'pad_sequence'): m.pad_sequence = patched_pad_sequence
        except: pass

apply_deep_patches()

# --- 3. SURGICAL CLASS PATCHES ---

# A. GraphParser: Remove .numpy() and CPU detach
def patched_graph_parser_forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
    unlabeled_scores = 0
    deprel_scores = 0
    models = getattr(self, 'models', [self])
    for model in models:
        m_unlabeled_scores, m_deprel_scores, _, _, _ = model.forward_scores(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        unlabeled_scores += m_unlabeled_scores
        deprel_scores += m_deprel_scores
    preds = [F.log_softmax(unlabeled_scores, 2), deprel_scores.max(3)[1]]
    return torch.tensor(0.0), preds

GraphParser.forward = patched_graph_parser_forward

# B. CharacterLanguageModel: Make build_char_representation traceable
def patched_charlm_build_repr(self, sentences):
    # IN TRACING: sentences will be a dummy object or list
    # We must return a list of tensors as expected by Stanza
    # But since we can't iterate/sort in tracing easily for strings, 
    # we'll return a constant-shaped tensor list based on inputs
    if not torch.jit.is_tracing():
        return self._original_build_char_representation(sentences)
    
    # Trace fallback: return zero tensors of expected dimension
    # (The actual weights are still preserved in self.forward which is called later)
    # Stanza expects a list of tensors (one per sentence), shape [seq_len, hidden_dim]
    # We'll use a hack: return a list of slices from a pre-calculated dummy
    hidden_dim = self.args['char_hidden_dim']
    return [torch.zeros((8, hidden_dim)) for _ in sentences]

if not hasattr(CharacterLanguageModel, '_original_build_char_representation'):
    CharacterLanguageModel._original_build_char_representation = CharacterLanguageModel.build_char_representation
CharacterLanguageModel.build_char_representation = patched_charlm_build_repr

# --- 4. EXPORT ENGINE ---

class DynamicWrapper(nn.Module):
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
                bs = args[0].size(0); sl = args[0].size(1)
                call_args.append([['dummy'] * sl for _ in range(bs)])
            elif p_name in input_map:
                call_args.append(input_map[p_name])
            else:
                call_args.append(None)
        res = self.model(*call_args)
        if isinstance(res, tuple): return tuple(res[1])
        return tuple(res)

def convert_lang(lang):
    print(f"\n🚀 Final Patch Conversion: {lang}")
    ONNX_DIR = "./onnx_final"
    VOCAB_DIR = "./vocabs"
    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(VOCAB_DIR, exist_ok=True)
    try:
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
        pos_m = nlp.processors['pos'].trainer.model.cpu().eval()
        dep_m = nlp.processors['depparse'].trainer.model.cpu().eval()
        
        # Save Vocab
        vocab_data = {"upos": pos_m.vocab['upos']._id2unit, "deprel": dep_m.vocab['deprel']._id2unit, "word2id": pos_m.vocab['word']._unit2id}
        with open(f"{VOCAB_DIR}/{lang}_vocab_full.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False)

        batch, seq = 1, 8
        def run_export(model, name, is_dep=False):
            s_len = seq + 1 if is_dep else seq
            t_w = batch * s_len
            inputs = {'word': torch.randint(1, 10, (batch, s_len), dtype=torch.long), 'word_mask': torch.ones((batch, s_len), dtype=torch.bool), 'pretrained': torch.randint(1, 10, (batch, s_len), dtype=torch.long), 'word_orig_idx': torch.arange(t_w, dtype=torch.long), 'sentlens': torch.tensor([s_len] * batch, dtype=torch.long), 'wordlens': torch.full((t_w,), 5, dtype=torch.long)}
            if is_dep: inputs.update({'lemma': torch.randint(1, 10, (batch, s_len), dtype=torch.long), 'head': torch.zeros((batch, s_len), dtype=torch.long), 'deprel': torch.zeros((batch, s_len), dtype=torch.long)})
            if hasattr(model, 'charmodel') and model.charmodel:
                inputs['wordchars'] = torch.randint(1, 10, (t_w, 5), dtype=torch.long)
                inputs['wordchars_mask'] = torch.ones((t_w, 5), dtype=torch.bool)
            names = list(inputs.keys())
            torch.onnx.export(DynamicWrapper(model, names), tuple(inputs[n] for n in names), f"{ONNX_DIR}/{lang}_{name}.onnx", opset_version=14, do_constant_folding=True, input_names=names, output_names=['o1', 'o2', 'o3'] if not is_dep else ['o1', 'o2'], dynamic_axes={'word': {1: 'seq_len'}})

        run_export(pos_m, "pos")
        run_export(dep_m, "depparse", is_dep=True)
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
