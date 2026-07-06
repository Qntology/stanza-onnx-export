import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.nn.utils.rnn import PackedSequence

# Extreme limits
sys.setrecursionlimit(100000)

# --- 1. PATCHES ---

class ONNXBilinear(nn.Module):
    def __init__(self, orig):
        super().__init__(); self.weight, self.bias = orig.weight, orig.bias
    def forward(self, x1, x2):
        if x1.dim() == 3: return torch.einsum('nli,oij,nlj->nlo', x1, self.weight, x2) + self.bias
        return torch.einsum('ni,oij,nj->no', x1, self.weight, x2) + self.bias

def patch_model_recursively(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Bilinear): setattr(module, name, ONNXBilinear(child))
        else: patch_model_recursively(child)

def apply_global_patches():
    import torch.nn.utils.rnn as rnn_utils
    rnn_utils.pack_padded_sequence = lambda input, lengths, batch_first=False, enforce_sorted=True: input if torch.jit.is_tracing() else rnn_utils.pack_padded_sequence(input, lengths, batch_first, enforce_sorted)
    rnn_utils.pad_packed_sequence = lambda sequence, batch_first=False, padding_value=0.0, total_length=None: (sequence, torch.tensor([sequence.size(1 if batch_first else 0)] * sequence.size(0 if batch_first else 1))) if torch.jit.is_tracing() else rnn_utils.pad_packed_sequence(sequence, batch_first, padding_value, total_length)
    import stanza.models.common.hlstm as hlstm
    import stanza.models.common.packed_lstm as packed_lstm
    def hlstm_forward(self, input, seqlens, hx=None):
        highway_func = (lambda x: x) if self.highway_func is None else self.highway_func
        hs, cs = [], []
        for l in range(self.num_layers):
            if self.dropout > 0 and l > 0: input = self.drop(input)
            layer_hx = (hx[0][l * self.num_directions:(l+1)*self.num_directions], hx[1][l * self.num_directions:(l+1)*self.num_directions]) if hx is not None else None
            if torch.jit.is_tracing(): layer_hx = None
            h, (ht, ct) = self.lstm[l].lstm(input, layer_hx)
            hs.append(ht); cs.append(ct)
            input = h + torch.sigmoid(self.gate[l](input)) * (self.highway[l](input) if self.highway_func is None else self.highway_func(self.highway[l](input)))
        return input, (torch.cat(hs, 0), torch.cat(cs, 0))
    hlstm.HighwayLSTM.forward = hlstm_forward
    packed_lstm.PackedLSTM.forward = lambda self, input, lengths, hx=None: self.lstm(input, None if torch.jit.is_tracing() else hx)

# --- 2. EXPORT ---

# import stanza  <-- 지우거나 주석 처리하세요!

def convert_lang(lang):
    # 언어별 하위 폴더 생성으로 경로 재지정
    ONNX_DIR = f"./onnx_final/{lang}"
    os.makedirs(ONNX_DIR, exist_ok=True)
    
    print(f"\n🚀 {lang}")
    try: 
        print(f"  ⬇️ Downloading from: https://huggingface.co/stanfordnlp/stanza-{lang}/tree/main/models")
        stanza.download(lang, processors='tokenize,pos,lemma,depparse')
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
    except Exception as e:
        print(f"    ❌ {e}")
        return

    # A. TOKENIZER
    if 'tokenize' in nlp.processors:
        try:
            print("  📦 Tokenizer...")
            m = nlp.processors['tokenize'].trainer.model.cpu().eval()
            in_size = m.rnn.input_size
            num_feats = in_size - 32
            inputs = (torch.zeros(1, 100).long(), torch.zeros(1, 100, num_feats).float(), torch.tensor([100]).long())
            
            # 모델 구성 및 텐서 값 확인을 위한 로그
            print(f"    [LOG] Tokenizer 모델 구성: rnn.input_size={in_size}, num_feats={num_feats}")
            print(f"    [LOG] Tokenizer 입력 텐서 Shape: x={inputs[0].shape}, f={inputs[1].shape}, l={inputs[2].shape}")
            
            # 파일 경로를 변경된 폴더 구조에 맞게 수정
            torch.onnx.export(m, inputs, f"{ONNX_DIR}/tokenizer.onnx", opset_version=14, 
                             input_names=['x','f','l'], output_names=['output'],
                             dynamic_axes={'x': {1: 'seq'}, 'f': {1: 'seq'}, 'output': {1: 'seq'}})
            print("    ✅ Done")
        except Exception as e: print(f"    ❌ {e}")

    # B. POS
    if 'pos' in nlp.processors:
        try:
            print("  📦 POS...")
            m = nlp.processors['pos'].trainer.model.cpu().eval()
            patch_model_recursively(m)
            class POSWrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, word, mask, chars, chars_mask, pre, oidx, slen, wlen):
                    ins = []
                    if self.m.args['word_emb_dim'] > 0: ins.append(self.m.word_emb(word))
                    if self.m.args['pretrain']: ins.append(self.m.trans_pretrained(self.m.pretrained_emb(pre)))
                    
                    def process_char(c_out):
                        c = c_out[0] if isinstance(c_out, tuple) else c_out
                        if c.dim() == 3:
                            if c.size(0) == chars.size(1) and c.size(1) == chars.size(0):
                                idx = (wlen - 1).clamp(min=0).unsqueeze(0).unsqueeze(2).expand(1, c.size(1), c.size(2))
                                c = c.gather(0, idx).squeeze(0)
                            elif c.size(1) == chars.size(1) and c.size(0) == chars.size(0):
                                idx = (wlen - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, c.size(2))
                                c = c.gather(1, idx).squeeze(1)
                        return c.reshape(word.size(0), word.size(1), -1)

                    if hasattr(self.m, 'charmodel_forward'):
                        cf = self.m.charmodel_forward(chars, chars_mask, wlen)
                        cb = self.m.charmodel_backward(chars, chars_mask, wlen)
                        ins.append(process_char(cf))
                        ins.append(process_char(cb))
                    elif hasattr(self.m, 'charmodel'):
                        try:
                            cm = self.m.charmodel(chars, chars_mask, wordlens=wlen, sentlens=slen)
                        except TypeError:
                            cm = self.m.charmodel(chars, chars_mask, wlen)
                        ins.append(process_char(cm))
                        
                    lstm_out, _ = self.m.taggerlstm(torch.cat(ins, 2), slen)
                    return self.m.upos_clf(F.relu(self.m.upos_hid(lstm_out)))
            
            inputs = (torch.randint(1, 10, (1, 8)), torch.ones((1, 8), dtype=torch.bool), 
                      torch.zeros(8, 32).long(), torch.ones(8, 32).bool(), 
                      torch.randint(1, 10, (1, 8)), torch.arange(8), torch.tensor([8]), torch.full((8,), 5))
            
            # 모델 구성 및 텐서 값 확인을 위한 로그
            print(f"    [LOG] POS 모델 구성: word_emb_dim={m.args['word_emb_dim']}, pretrain={m.args['pretrain']}")
            print(f"    [LOG] POS 입력 텐서 Shape: word={inputs[0].shape}, mask={inputs[1].shape}, pre={inputs[4].shape}, slen={inputs[6].shape}")
            
            # 파일 경로를 변경된 폴더 구조에 맞게 수정
            torch.onnx.export(POSWrapper(m), inputs, f"{ONNX_DIR}/pos.onnx", opset_version=14,
                             input_names=['word','mask','chars','chars_mask','pre','oidx','slen','wlen'],
                             output_names=['upos'],
                             dynamic_axes={'word': {1: 'seq'}, 'mask': {1: 'seq'}, 'pre': {1: 'seq'}, 'upos': {1: 'seq'}})
            print("    ✅ Done")
        except Exception as e: print(f"    ❌ {e}")

    # C. DEPPARSE
    if 'depparse' in nlp.processors:
        try:
            print("  📦 Depparse...")
            m = nlp.processors['depparse'].trainer.model.cpu().eval()
            patch_model_recursively(m)
            class DepWrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, word, mask, chars, chars_mask, upos, pre, lemma, slen, wlen):
                    ins = []
                    if self.m.args['pretrain']: ins.append(self.m.trans_pretrained(self.m.pretrained_emb(pre)))
                    if self.m.args['word_emb_dim'] > 0:
                        ins.append(self.m.word_emb(word))
                        ins.append(self.m.lemma_emb(lemma))
                    if self.m.args['tag_emb_dim'] > 0:
                        p_emb = self.m.upos_emb(upos)
                        if hasattr(self.m, 'xpos_emb'):
                            if isinstance(self.m.xpos_emb, nn.ModuleList):
                                for l in self.m.xpos_emb: p_emb = p_emb + l(upos)
                            else: p_emb = p_emb + self.m.xpos_emb(upos)
                        ins.append(p_emb)
                        
                    def process_char(c_out):
                        c = c_out[0] if isinstance(c_out, tuple) else c_out
                        if c.dim() == 3:
                            if c.size(0) == chars.size(1) and c.size(1) == chars.size(0):
                                idx = (wlen - 1).clamp(min=0).unsqueeze(0).unsqueeze(2).expand(1, c.size(1), c.size(2))
                                c = c.gather(0, idx).squeeze(0)
                            elif c.size(1) == chars.size(1) and c.size(0) == chars.size(0):
                                idx = (wlen - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, c.size(2))
                                c = c.gather(1, idx).squeeze(1)
                        return c.reshape(word.size(0), word.size(1), -1)

                    if hasattr(self.m, 'charmodel_forward'):
                        cf = self.m.charmodel_forward(chars, chars_mask, wlen)
                        cb = self.m.charmodel_backward(chars, chars_mask, wlen)
                        ins.append(torch.cat([process_char(cf), process_char(cb)], dim=2))
                    elif hasattr(self.m, 'charmodel'):
                        try:
                            cm = self.m.charmodel(chars, chars_mask, wordlens=wlen, sentlens=slen)
                        except TypeError:
                            cm = self.m.charmodel(chars, chars_mask, wlen)
                        ins.append(process_char(cm))
                    
                    curr = torch.cat(ins, 2)
                    if curr.size(-1) < self.m.parserlstm.input_size:
                        curr = torch.cat([curr, torch.zeros(curr.size(0), curr.size(1), self.m.parserlstm.input_size - curr.size(-1))], 2)
                    
                    lstm_out, _ = self.m.parserlstm(curr, slen)
                    if hasattr(self.m, 'unlabeled'): 
                        return self.m.unlabeled(lstm_out, lstm_out).squeeze(3), self.m.deprel(lstm_out, lstm_out)
                    arc = self.m.arc_embedding(lstm_out, lstm_out)
                    return self.m.unlabeled_linear(arc).squeeze(3), self.m.deprel_linear(arc)
            
            inputs = (torch.randint(1, 10, (1, 9)), torch.ones((1, 9), dtype=torch.bool), 
                      torch.zeros(9, 32).long(), torch.ones(9, 32).bool(),
                      torch.zeros(1, 9).long(), torch.randint(1, 10, (1, 9)), 
                      torch.randint(1, 10, (1, 9)), torch.tensor([9]), torch.full((9,), 5))
            
            # 모델 구성 및 텐서 값 확인을 위한 로그
            print(f"    [LOG] Depparse 모델 구성: parserlstm.input_size={m.parserlstm.input_size}, tag_emb_dim={m.args['tag_emb_dim']}")
            print(f"    [LOG] Depparse 입력 텐서 Shape: word={inputs[0].shape}, mask={inputs[1].shape}, chars={inputs[2].shape}, chars_mask={inputs[3].shape}, upos={inputs[4].shape}, pre={inputs[5].shape}, lemma={inputs[6].shape}, slen={inputs[7].shape}, wlen={inputs[8].shape}")
            
            # 파일 경로를 변경된 폴더 구조에 맞게 수정
            torch.onnx.export(DepWrapper(m), inputs, f"{ONNX_DIR}/depparse.onnx", opset_version=14,
                             input_names=['word','mask','chars','chars_mask','upos','pre','lemma','slen','wlen'],
                             output_names=['unlabeled','deprel'],
                             dynamic_axes={'word': {1: 'seq'}, 'mask': {1: 'seq'}, 'chars': {0: 'seq'}, 'chars_mask': {0: 'seq'}, 'upos': {1: 'seq'}, 'pre': {1: 'seq'}, 'lemma': {1: 'seq'}, 'unlabeled': {1: 'seq', 2: 'seq'}, 'deprel': {1: 'seq', 2: 'seq'}, 'wlen': {0: 'seq'}})
            print("    ✅ Done")
        except Exception as e: print(f"    ❌ {e}")

    # D. VOCAB EXTRACTION
    try:
        print("  📦 Vocab Extraction...")
        vocab_data = {}
        for proc_name in ['tokenize', 'pos', 'lemma', 'depparse']:
            if proc_name in nlp.processors:
                proc = nlp.processors[proc_name]
                if hasattr(proc, 'vocab'):
                    vocab_data[proc_name] = {}
                    v = proc.vocab
                    if hasattr(v, '_vocabs'): # MultiVocab 지원 프로세서
                        for k, sub_v in v._vocabs.items():
                            if hasattr(sub_v, '_id2unit'):
                                vocab_data[proc_name][k] = sub_v._id2unit
                    elif hasattr(v, '_id2unit'): # Single Vocab 프로세서
                        vocab_data[proc_name]['main'] = v._id2unit
        
        if vocab_data:
            vocab_path = f"{ONNX_DIR}/vocab.json"
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            # Vocab 구성 및 저장 관련 확인 로그 추가
            print(f"    [LOG] Vocab 구성 확인: 포함된 프로세서={list(vocab_data.keys())}")
            print(f"    [LOG] Vocab 저장 경로: {vocab_path}")
            print("    ✅ Done")
    except Exception as e:
        print(f"    ❌ {e}")

if __name__ == "__main__":
    # 1. 패치를 최우선으로 적용하여 PyTorch 내부 함수를 덮어씌웁니다.
    apply_global_patches()
    
    # 2. 패치가 완료된 후 stanza를 임포트해야 패치된 함수를 가져다 씁니다.
    import stanza 
    
    # 3. 변환 작업을 시작합니다.
    for l in ['en', 'de', 'es', 'fr', 'ja', 'pt', 'ar', 'cs', 'it', 'ko', 'nl', 'zh-hans']: 
        convert_lang(l)
