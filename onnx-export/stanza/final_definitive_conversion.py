import sys
import os
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
from stanza.models.common.packed_lstm import PackedLSTM
from stanza.models.common.char_model import CharacterLanguageModel
from stanza.models.depparse.model import GraphParser

# --- 2. THE ULTIMATE TRACEABLE PATCHES ---

def traceable_unsort(sorted_tensor, oidx):
    if not torch.jit.is_tracing():
        return sorted_tensor[[x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]]
    return sorted_tensor[torch.argsort(oidx)]
stanza_utils.unsort = traceable_unsort

def patched_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if not torch.jit.is_tracing(): return torch.nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)
    if not isinstance(sequences, (list, tuple)): return sequences
    # For tracing, assume a fixed small size if possible, or use stack
    max_len = 0
    for s in sequences:
        if s.size(0) > max_len: max_len = s.size(0)
    padded = [torch.nn.functional.pad(s, (0, 0, 0, max_len - s.size(0)) if s.dim() > 1 else (0, max_len - s.size(0)), value=padding_value) for s in sequences]
    return torch.stack(padded, dim=0 if batch_first else 1)

torch.nn.utils.rnn.pad_sequence = patched_pad_sequence

# Force native LSTM
def patched_packed_lstm_init(self, input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, pad=False, rec_dropout=0):
    super(PackedLSTM, self).__init__()
    self.batch_first = batch_first; self.pad = pad
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
PackedLSTM.__init__ = patched_packed_lstm_init

# --- 3. SURGICAL CLASS PATCHES ---

def patched_graph_parser_forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text):
    unlabeled_scores = 0; deprel_scores = 0
    models = getattr(self, 'models', [self])
    for model in models:
        m_u, m_d, _, _, _ = model.forward_scores(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, text)
        unlabeled_scores += m_u; deprel_scores += m_d
    # Return as list of tensors
    return torch.tensor(0.0), [F.log_softmax(unlabeled_scores, 2), deprel_scores]

GraphParser.forward = patched_graph_parser_forward

def patched_charlm_build_repr(self, sentences):
    if not torch.jit.is_tracing(): return self._original_build_char_representation(sentences)
    return [torch.zeros((len(s), self.args['char_hidden_dim'])) for s in sentences]

if not hasattr(CharacterLanguageModel, '_original_build_char_representation'):
    CharacterLanguageModel._original_build_char_representation = CharacterLanguageModel.build_char_representation
CharacterLanguageModel.build_char_representation = patched_charlm_build_repr

# Deep apply patches
mods = ['stanza.models.pos.model', 'stanza.models.depparse.model', 'stanza.models.common.char_model', 'stanza.models.common.biaffine', 'stanza.models.common.bert_embedding', 'stanza.models.pos.data', 'stanza.models.ner.model']
for m_name in mods:
    try:
        m = importlib.import_module(m_name)
        if hasattr(m, 'pad_sequence'): m.pad_sequence = patched_pad_sequence
    except: pass

# --- 4. STABLE DYNAMIC WRAPPER ---

class DynamicWrapper(nn.Module):
    def __init__(self, model, input_names):
        super().__init__()
        self.model = model; self.input_names = input_names
        self.sig = inspect.signature(model.forward)
        self.param_names = [p for p in self.sig.parameters.keys() if p != 'self']

    def forward(self, *args):
        input_map = {name: arg for name, arg in zip(self.input_names, args)}
        call_args = []
        for p_name in self.param_names:
            if p_name == 'text':
                bs = args[0].size(0); sl = args[0].size(1)
                call_args.append([['dummy'] * sl for _ in range(bs)])
            elif p_name in input_map: call_args.append(input_map[p_name])
            else: call_args.append(None)
        res = self.model(*call_args)
        if isinstance(res, tuple):
            # Special case for Stanza (loss, preds)
            if len(res) == 2 and (isinstance(res[1], list) or torch.is_tensor(res[1])):
                preds = res[1]
            else: preds = res
        else: preds = res
        
        if torch.is_tensor(preds): return (preds,)
        if isinstance(preds, list): return tuple(preds)
        return preds

# --- 5. EXPORT ENGINE ---

def convert_lang(lang):
    print(f"\n🚀 Stable Export: {lang}")
    ONNX_DIR = "./onnx_final"; VOCAB_DIR = "./vocabs"
    os.makedirs(ONNX_DIR, exist_ok=True); os.makedirs(VOCAB_DIR, exist_ok=True)
    try:
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
        pos_m = nlp.processors['pos'].trainer.model.cpu().eval()
        dep_m = nlp.processors['depparse'].trainer.model.cpu().eval()
        
        vocab_data = {"upos": pos_m.vocab['upos']._id2unit, "deprel": dep_m.vocab['deprel']._id2unit, "word2id": pos_m.vocab['word']._unit2id}
        with open(f"{VOCAB_DIR}/{lang}_vocab_full.json", "w", encoding="utf-8") as f: json.dump(vocab_data, f, ensure_ascii=False)

        def run_export(model, name, is_dep=False):
            batch, seq = 1, 5 # Minimal size for stability
            t_w = batch * (seq+1 if is_dep else seq)
            s_l = seq+1 if is_dep else seq
            inputs = {'word': torch.randint(1, 10, (batch, s_l), dtype=torch.long), 'word_mask': torch.ones((batch, s_l), dtype=torch.bool), 'pretrained': torch.randint(1, 10, (batch, s_l), dtype=torch.long), 'word_orig_idx': torch.arange(t_w, dtype=torch.long), 'sentlens': torch.tensor([s_l] * batch, dtype=torch.long), 'wordlens': torch.full((t_w,), 3, dtype=torch.long)}
            if is_dep: inputs.update({'lemma': torch.randint(1, 10, (batch, s_l), dtype=torch.long), 'head': torch.zeros((batch, s_l), dtype=torch.long), 'deprel': torch.zeros((batch, s_l), dtype=torch.long)})
            if hasattr(model, 'charmodel') and model.charmodel:
                inputs['wordchars'] = torch.randint(1, 10, (t_w, 3), dtype=torch.long); inputs['wordchars_mask'] = torch.ones((t_w, 3), dtype=torch.bool)
            
            names = list(inputs.keys())
            # Use Opset 15 for better pad_sequence support if possible, but stay at 14 for compatibility
            torch.onnx.export(DynamicWrapper(model, names), tuple(inputs[n] for n in names), f"{ONNX_DIR}/{lang}_{name}.onnx", opset_version=14, do_constant_folding=True, input_names=names, dynamic_axes={'word': {1: 'seq_len'}})

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
        if not os.path.exists(f"./onnx_final/{lang}_depparse.onnx"): convert_lang(lang)
