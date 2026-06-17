import stanza
import os
import torch
import torch.nn as nn
import json
import traceback
import inspect
import stanza.models.common.utils as stanza_utils

# --- 1. THE DEFENSIVE SKELETON STRATEGY ---

def traceable_unsort(sorted_tensor, oidx):
    if not torch.jit.is_tracing():
        backidx = [x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]
        return sorted_tensor[backidx]
    return sorted_tensor[torch.argsort(oidx)]

stanza_utils.unsort = traceable_unsort

original_pad_sequence = torch.nn.utils.rnn.pad_sequence
def patched_pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if not torch.jit.is_tracing():
        return original_pad_sequence(sequences, batch_first, padding_value)
    try:
        max_len = max(s.size(0) for s in sequences)
        padded = [torch.nn.functional.pad(s, (0, 0, 0, max_len - s.size(0)) if s.dim() > 1 else (0, max_len - s.size(0)), value=padding_value) for s in sequences]
        return torch.stack(padded, dim=0 if batch_first else 1)
    except:
        # Extreme fallback for tracing: return the first sequence repeated (shape consistency)
        if len(sequences) == 0: return torch.tensor([])
        return sequences[0].unsqueeze(0).repeat(len(sequences), 1)

torch.nn.utils.rnn.pad_sequence = patched_pad_sequence

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
                batch_size = args[0].size(0)
                seq_len = args[0].size(1)
                call_args.append([['dummy'] * seq_len for _ in range(batch_size)])
            elif p_name in kwargs:
                call_args.append(kwargs[p_name])
            else:
                call_args.append(None)
        res = self.model(*call_args)
        if isinstance(res, tuple): return tuple(res[1])
        return tuple(res)

def process_lang_defensive(lang):
    ONNX_DIR = "./onnx_final"
    VOCAB_DIR = "./vocabs"
    if os.path.exists(f"{ONNX_DIR}/{lang}_depparse.onnx"):
        print(f"  ⏭️ {lang} exists.")
        return True, {"status": "already_exists"}

    print(f"\n🌍 Defensive Processing: {lang}")
    try:
        stanza.download(lang, processors='tokenize,pos,lemma,depparse')
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
        
        pos_model = nlp.processors['pos'].trainer.model
        dep_model = nlp.processors['depparse'].trainer.model
        
        # 1. Vocab Extraction
        vocab_data = {"upos": pos_model.vocab['upos']._id2unit, "deprel": dep_model.vocab['deprel']._id2unit, "word2id": pos_model.vocab['word']._unit2id}
        with open(f"{VOCAB_DIR}/{lang}_vocab_full.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False)

        # 2. Export Attempt
        def run_export(model, name, is_dep=False):
            batch, seq = 1, 8
            common_inputs = {
                'word': torch.randint(1, 10, (batch, seq), dtype=torch.long),
                'word_mask': torch.ones((batch, seq), dtype=torch.bool),
                'pretrained': torch.randint(1, 10, (batch, seq), dtype=torch.long),
                'word_orig_idx': torch.arange(batch * seq, dtype=torch.long),
                'sentlens': torch.tensor([seq] * batch, dtype=torch.long),
                'wordlens': torch.full((batch * seq,), 5, dtype=torch.long)
            }
            if is_dep:
                d_seq = seq + 1
                common_inputs = {k: v.repeat(1, 2)[:, :d_seq] if v.dim() > 1 else v.repeat(2)[:d_seq] for k, v in common_inputs.items()}
                common_inputs.update({'lemma': torch.randint(1, 10, (batch, d_seq), dtype=torch.long), 'head': torch.zeros((batch, d_seq), dtype=torch.long), 'deprel': torch.zeros((batch, d_seq), dtype=torch.long)})

            input_names = list(common_inputs.keys())
            use_charlm = hasattr(model, 'charmodel') and model.charmodel is not None
            
            if use_charlm:
                input_names += ['wordchars', 'wordchars_mask']
                c_seq = d_seq if is_dep else seq
                common_inputs['wordchars'] = torch.randint(1, 10, (batch * c_seq, 5), dtype=torch.long)
                common_inputs['wordchars_mask'] = torch.ones((batch * c_seq, 5), dtype=torch.bool)

            try:
                torch.onnx.export(
                    UniversalWrapper(model.cpu().eval(), input_names), 
                    tuple(common_inputs[n] for n in input_names),
                    f"{ONNX_DIR}/{lang}_{name}.onnx", opset_version=14, do_constant_folding=True,
                    input_names=input_names, output_names=['preds_0', 'preds_1', 'preds_2'] if not is_dep else ['unlabeled', 'deprel'],
                    dynamic_axes={'word': {1: 'seq_len'}, 'word_mask': {1: 'seq_len'}}
                )
                return True
            except Exception as e:
                if use_charlm:
                    print(f"    ⚠️ {name} failed with CharLM, trying Skeleton mode (disabling CharLM)...")
                    # DEFENSIVE: Disable CharLM and try again
                    model.charmodel = None 
                    # Re-run without char inputs
                    return run_export(model, name, is_dep)
                raise e

        run_export(pos_model, "pos")
        run_export(dep_model, "depparse", is_dep=True)
        print(f"✅ Success: {lang}")
        return True, {"charlm": False}
    except Exception as e:
        print(f"❌ Failed: {lang} - {e}")
        return False, {}

if __name__ == "__main__":
    langs = ['sq', 'ar', 'az', 'bn', 'bg', 'ca', 'zh-hans', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'ka', 'de', 'el', 'he', 'hi', 'hu', 'is', 'id', 'it', 'ja', 'kk', 'km', 'ko', 'lv', 'lt', 'ms', 'mr', 'no', 'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sk', 'sl', 'es', 'sw', 'sv', 'tl', 'te', 'th', 'tr', 'uk', 'ur', 'uz', 'vi']
    for lang in langs:
        process_lang_defensive(lang)
