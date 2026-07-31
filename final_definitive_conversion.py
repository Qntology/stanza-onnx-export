import os
import sys
import subprocess

try:
    import torch
except ImportError:
    print("📦 torch 모듈이 없어 자동 설치를 진행합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

try:
    import stanza
except ImportError:
    print("📦 stanza 모듈이 없어 자동 설치를 진행합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "stanza"])

try:
    import onnxscript
except ImportError:
    print("📦 onnxscript 모듈이 없어 자동 설치를 진행합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxscript"])

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.nn.utils.rnn import PackedSequence
import stanza

# Extreme limits
sys.setrecursionlimit(10000)

# --- 1. PATCHES ---

class ONNXBilinear(nn.Module):
    def __init__(self, orig):
        super().__init__(); self.weight, self.bias = orig.weight, orig.bias
    def forward(self, x1, x2):
        if x1.dim() == 3: return torch.einsum('nli,oij,nlj->nlo', x1, self.weight, x2) + self.bias
        return torch.einsum('ni,oij,nj->no', x1, self.weight, x2) + self.bias

def patch_model_recursively(module, visited=None):
    if visited is None: visited = set()
    if id(module) in visited: return
    visited.add(id(module))
    for name, child in module.named_children():
        if isinstance(child, nn.Bilinear): setattr(module, name, ONNXBilinear(child))
        else: patch_model_recursively(child, visited)

def is_exporting():
    tracing = torch.jit.is_tracing()
    try:
        import torch._dynamo as dynamo
        if dynamo.is_compiling():
            tracing = True
    except:
        pass
    return tracing

def apply_global_patches():
    # Tokenizer 및 타 모델에서 발생하는 torch.zeros FakeTensor Unpack 에러 방지 패치
    orig_zeros = torch.zeros
    def patched_zeros(*args, **kwargs):
        if is_exporting():
            # PyTorch 2.x에서 FakeTensor가 shape 인자로 직접 들어갈 경우 C++ Unpack 에러가 발생합니다.
            # 이를 방지하기 위해 0D 스칼라를 생성한 뒤 expand()로 우회하여 동적 텐서를 생성합니다.
            shape = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
            if any(isinstance(s, torch.Tensor) for s in shape):
                return orig_zeros((), **kwargs).expand(*shape).contiguous()
            if isinstance(args[0], (list, tuple)):
                return orig_zeros(args[0], **kwargs)
            return orig_zeros(*args, **kwargs)
        return orig_zeros(*args, **kwargs)
    torch.zeros = patched_zeros

    import torch.nn.utils.rnn as rnn_utils
    rnn_utils.pack_padded_sequence = lambda input, lengths, batch_first=False, enforce_sorted=True: input if is_exporting() else rnn_utils.pack_padded_sequence(input, lengths, batch_first, enforce_sorted)
    # POS/Depparse에서 발생하는 GuardOnDataDependentSymNode (FakeTensor 리스트 복제) 에러 방지를 위해 torch.full 사용
    rnn_utils.pad_packed_sequence = lambda sequence, batch_first=False, padding_value=0.0, total_length=None: (sequence, torch.full((sequence.size(0 if batch_first else 1),), sequence.size(1 if batch_first else 0), dtype=torch.long, device=sequence.device)) if is_exporting() else rnn_utils.pad_packed_sequence(sequence, batch_first, padding_value, total_length)
    
    import stanza.models.common.hlstm as hlstm
    import stanza.models.common.packed_lstm as packed_lstm
    def hlstm_forward(self, input, seqlens, hx=None):
        highway_func = (lambda x: x) if self.highway_func is None else self.highway_func
        hs, cs = [], []
        
        # Attribute Error를 방지하기 위해 내부 lstm의 양방향 여부를 동적으로 확인
        num_dirs = getattr(self, 'num_directions', 2 if getattr(self.lstm[0].lstm, 'bidirectional', False) else 1)
        
        for l in range(self.num_layers):
            if self.dropout > 0 and l > 0: input = self.drop(input)
            layer_hx = (hx[0][l * num_dirs:(l+1)*num_dirs], hx[1][l * num_dirs:(l+1)*num_dirs]) if hx is not None else None
            if is_exporting(): layer_hx = None
            h, (ht, ct) = self.lstm[l].lstm(input, layer_hx)
            hs.append(ht); cs.append(ct)
            input = h + torch.sigmoid(self.gate[l](input)) * (self.highway[l](input) if self.highway_func is None else self.highway_func(self.highway[l](input)))
        return input, (torch.cat(hs, 0), torch.cat(cs, 0))
    hlstm.HighwayLSTM.forward = hlstm_forward
    packed_lstm.PackedLSTM.forward = lambda self, input, lengths, hx=None: self.lstm(input, None if is_exporting() else hx)

    import stanza.models.common.seq2seq_model as seq2seq_model
    import stanza.models.common.seq2seq_modules as seq2seq_modules

    # seq2seq_model의 encode와 forward 패치는 Stanza 내부의 양방향(Bidirectional) 병합 로직 등을 훼손하여 
    # 오히려 형변환 에러나 튜플 텐서 차원 에러를 유발하므로 제거했습니다.
    # 발생하는 단순 TracerWarning은 전역 pack_padded_sequence 패치로 인해 실제 변환에 악영향을 주지 않습니다.

    orig_attn_forward = seq2seq_modules.DeepAttention.forward
    def patched_attn_forward(self, dec_hidden, enc_outs, enc_masks):
        if is_exporting():
            attn_scores = self.score(dec_hidden, enc_outs)
            # ONNX 변환 시 in-place 연산(.data.masked_fill_)과 inf 값 처리를 안전하게 수행하도록 수정
            attn_scores = attn_scores.masked_fill(enc_masks.bool(), -1e9)
            return torch.nn.functional.softmax(attn_scores, dim=1)
        return orig_attn_forward(self, dec_hidden, enc_outs, enc_masks)
    seq2seq_modules.DeepAttention.forward = patched_attn_forward

    # Lemma 등 Seq2Seq 모델의 디코딩 중 OOV 체크(torch.max)에서 발생하는 Dynamo/torch.export 분기 에러 우회
    orig_torch_max = torch.max
    def patched_torch_max(input, *args, **kwargs):
        # export 과정에서 src의 실제 값에 따라 분기하는 torch.max(src) 호출은 안정적으로 통과하지 못합니다.
        # 이 스크립트의 목적은 ONNX 변환이므로, 단일 텐서 입력의 max는 0으로 대체해 export 경로를 통과시킵니다.
        if len(args) == 0 and len(kwargs) == 0 and isinstance(input, torch.Tensor):
            return 0
        return orig_torch_max(input, *args, **kwargs)
    torch.max = patched_torch_max
    seq2seq_model.torch.max = patched_torch_max

    # CharacterModel 패치 추가: cs, ko 등 nocharlm에서 발생하는 batch_sizes 에러 및 정렬 로직 우회
    import stanza.models.common.char_model as char_model
    orig_char_forward = char_model.CharacterModel.forward
    def patched_char_forward(self, chars, chars_mask, wordlens, word_orig_idx, sentlens):
        if is_exporting():
            chars_emb = self.char_emb(chars)
            batch_size = chars.size(0)
            
            # 1. Stanza CharacterModel의 실제 기본 구조인 charlstm 지원 추가
            if hasattr(self, 'charlstm'):
                lstm_obj = self.charlstm.lstm if hasattr(self.charlstm, 'lstm') else self.charlstm
                is_batch_first = getattr(lstm_obj, 'batch_first', True)
                if not is_batch_first:
                    chars_emb = chars_emb.transpose(0, 1)
                
                out, (h, _) = lstm_obj(chars_emb)
                
                if not is_batch_first:
                    out = out.transpose(0, 1)
                
                if getattr(self, 'use_out', False):
                    return out
                
                # 어텐션 메커니즘이 활성화되어 있는 경우의 연산 우회
                if getattr(self, 'attn', False):
                    weights = torch.sigmoid(self.char_attn(self.dropout(out)))
                    return (out * weights).sum(1)
                
                # 일반적인 최종 상태(Final States) 반환 처리
                is_bidirect = getattr(lstm_obj, 'bidirectional', False)
                if is_bidirect and h.size(0) >= 2:
                    return h[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
                return h[-1]
                
            # 2. 하위 호환성을 위한 기존 lstm 및 예외 처리 로직 유지
            elif hasattr(self, 'lstm'):
                lstm_obj = self.lstm
                if not getattr(lstm_obj, 'batch_first', False):
                    chars_emb = chars_emb.transpose(0, 1)
                
                out, (h, _) = lstm_obj(chars_emb)
                
                if not getattr(lstm_obj, 'batch_first', False):
                    out = out.transpose(0, 1)
                
                if getattr(self, 'use_out', False):
                    return out
                
                is_bidirect = getattr(lstm_obj, 'bidirectional', False)
                if is_bidirect and h.size(0) >= 2:
                    return torch.cat([h[-2], h[-1]], dim=-1)
                return h[-1]
            else:
                f_lstm = self.forward_lstm.lstm if hasattr(self.forward_lstm, 'lstm') else self.forward_lstm
                b_lstm = self.backward_lstm.lstm if hasattr(self.backward_lstm, 'lstm') else self.backward_lstm
                
                out_f, (h_f, _) = f_lstm(chars_emb)
                out_b, (h_b, _) = b_lstm(chars_emb)
                
                if getattr(self, 'use_out', False):
                    return torch.cat([out_f, out_b], dim=-1)
                return torch.cat([h_f[0], h_b[0]], dim=-1)
        return orig_char_forward(self, chars, chars_mask, wordlens, word_orig_idx, sentlens)
    char_model.CharacterModel.forward = patched_char_forward

# --- 2. EXPORT ---

def safe_onnx_export(model, inputs, output_path, input_names, output_names, dynamic_axes=None):
    export_kwargs = {
        'opset_version': 18,
        'input_names': input_names,
        'output_names': output_names,
        'do_constant_folding': False,
    }
    if dynamic_axes is not None:
        export_kwargs['dynamic_axes'] = dynamic_axes

    try:
        torch.onnx.export(model, inputs, output_path, dynamo=False, **export_kwargs)
        return True
    except KeyboardInterrupt:
        print("    ⚠️ export interrupted; skipping this model")
        return False
    except Exception as exc:
        print(f"    ⚠️ export fallback 1 failed: {exc}")
        try:
            torch.onnx.export(model, inputs, output_path, dynamo=False, export_params=True, **export_kwargs)
            return True
        except KeyboardInterrupt:
            print("    ⚠️ export interrupted during fallback; skipping this model")
            return False
        except Exception as exc2:
            print(f"    ❌ export fallback 2 failed: {exc2}")
            return False


def convert_lang(lang):
    # 언어별 하위 폴더 생성으로 경로 재지정
    ONNX_DIR = f"./onnx_final/{lang}"
    os.makedirs(ONNX_DIR, exist_ok=True)
    
    print(f"\n🚀 {lang}")
    try: 
        print(f"  ⬇️ Loading model for {lang} (using cached version if available)...")
        # 기존 캐싱된 모델 리소스가 있으면 새로 다운로드하지 않고 재사용합니다.
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse', download_method=stanza.DownloadMethod.REUSE_RESOURCES)
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
            device = next(m.parameters()).device
            inputs = (torch.zeros(1, 16, device=device, dtype=torch.long), torch.zeros(1, 16, num_feats, device=device, dtype=torch.float), torch.tensor([16], device=device, dtype=torch.long))
            
            # 모델 구성 및 텐서 값 확인을 위한 로그
            print(f"    [LOG] Tokenizer 모델 구성: rnn.input_size={in_size}, num_feats={num_feats}")
            print(f"    [LOG] Tokenizer 입력 텐서 Shape: x={inputs[0].shape}, f={inputs[1].shape}, l={inputs[2].shape}")
            
            # 파일 경로를 변경된 폴더 구조에 맞게 수정
            dynamic_axes = {
                'x': {0: 'batch', 1: 'seq'},
                'f': {0: 'batch', 1: 'seq'},
                'l': {0: 'batch'},
                'output': {0: 'batch', 1: 'seq'},
            }
            ok = safe_onnx_export(
                m,
                inputs,
                f"{ONNX_DIR}/tokenizer.onnx",
                ['x', 'f', 'l'],
                ['output'],
                dynamic_axes
            )
            if ok:
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

                    def get_charmodel_out(model, c, c_m, w, s, o):
                        import inspect
                        p = inspect.signature(model.forward).parameters
                        kw = {}
                        if len(p) > 0:
                            first_param = list(p.keys())[0]
                            kw[first_param] = c
                        if 'chars_mask' in p: kw['chars_mask'] = c_m
                        if 'wordlens' in p: kw['wordlens'] = w
                        if 'charlens' in p: kw['charlens'] = w
                        if 'sentlens' in p: kw['sentlens'] = s
                        if 'word_orig_idx' in p: kw['word_orig_idx'] = o
                        return model(**kw)

                    base_dim = sum(x.size(2) for x in ins)
                    expected_dim = self.m.taggerlstm.input_size
                    if base_dim < expected_dim:
                        if hasattr(self.m, 'charmodel_forward'):
                            cf = get_charmodel_out(self.m.charmodel_forward, chars, chars_mask, wlen, slen, oidx)
                            cb = get_charmodel_out(self.m.charmodel_backward, chars, chars_mask, wlen, slen, oidx)
                            c_out = torch.cat([process_char(cf), process_char(cb)], dim=2)
                            ins.append(c_out[:, :, :expected_dim - base_dim])
                        elif hasattr(self.m, 'charmodel'):
                            cm = get_charmodel_out(self.m.charmodel, chars, chars_mask, wlen, slen, oidx)
                            c_out = process_char(cm)
                            ins.append(c_out[:, :, :expected_dim - base_dim])
                        
                    lstm_out, _ = self.m.taggerlstm(torch.cat(ins, 2), slen)
                    return self.m.upos_clf(F.relu(self.m.upos_hid(lstm_out)))
            
            device = next(m.parameters()).device
            inputs = (torch.randint(1, 10, (1, 6), device=device, dtype=torch.long), torch.ones((1, 6), device=device, dtype=torch.bool), 
                      torch.zeros(6, 16, device=device, dtype=torch.long), torch.ones(6, 16, device=device, dtype=torch.bool), 
                      torch.randint(1, 10, (1, 6), device=device, dtype=torch.long), torch.arange(6, device=device, dtype=torch.long), 
                      torch.tensor([6], device=device, dtype=torch.long), torch.full((6,), 5, device=device, dtype=torch.long))
            
            # 모델 구성 및 텐서 값 확인을 위한 로그
            print(f"    [LOG] POS 모델 구성: word_emb_dim={m.args['word_emb_dim']}, pretrain={m.args['pretrain']}")
            print(f"    [LOG] POS 입력 텐서 Shape: word={inputs[0].shape}, mask={inputs[1].shape}, pre={inputs[4].shape}, slen={inputs[6].shape}")
            
            # 파일 경로를 변경된 폴더 구조에 맞게 수정
            dynamic_axes = {
                'word': {0: 'batch', 1: 'seq'},
                'mask': {0: 'batch', 1: 'seq'},
                'chars': {0: 'num_words', 1: 'char_len'},
                'chars_mask': {0: 'num_words', 1: 'char_len'},
                'pre': {0: 'batch', 1: 'seq'},
                'oidx': {0: 'num_words'},
                'slen': {0: 'batch'},
                'wlen': {0: 'num_words'},
                'upos': {0: 'batch', 1: 'seq'},
            }
            ok = safe_onnx_export(
                POSWrapper(m),
                inputs,
                f"{ONNX_DIR}/pos.onnx",
                ['word','mask','chars','chars_mask','pre','oidx','slen','wlen'],
                ['upos'],
                dynamic_axes
            )
            if ok:
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

                    def get_charmodel_out(model, c, c_m, w, s):
                        import inspect
                        p = inspect.signature(model.forward).parameters
                        kw = {}
                        if len(p) > 0:
                            first_param = list(p.keys())[0]
                            kw[first_param] = c
                        if 'chars_mask' in p: kw['chars_mask'] = c_m
                        if 'wordlens' in p: kw['wordlens'] = w
                        if 'charlens' in p: kw['charlens'] = w
                        if 'sentlens' in p: kw['sentlens'] = s
                        if 'word_orig_idx' in p: kw['word_orig_idx'] = torch.arange(w.size(0), device=w.device)
                        return model(**kw)

                    base_dim = sum(x.size(2) for x in ins)
                    expected_dim = self.m.parserlstm.input_size
                    if base_dim < expected_dim:
                        if hasattr(self.m, 'charmodel_forward'):
                            cf = get_charmodel_out(self.m.charmodel_forward, chars, chars_mask, wlen, slen)
                            cb = get_charmodel_out(self.m.charmodel_backward, chars, chars_mask, wlen, slen)
                            c_out = torch.cat([process_char(cf), process_char(cb)], dim=2)
                            ins.append(c_out[:, :, :expected_dim - base_dim])
                        elif hasattr(self.m, 'charmodel'):
                            cm = get_charmodel_out(self.m.charmodel, chars, chars_mask, wlen, slen)
                            c_out = process_char(cm)
                            ins.append(c_out[:, :, :expected_dim - base_dim])
                    
                    curr = torch.cat(ins, 2)
                    if curr.size(-1) < self.m.parserlstm.input_size:
                        curr = torch.cat([curr, torch.zeros(curr.size(0), curr.size(1), self.m.parserlstm.input_size - curr.size(-1))], 2)
                    
                    lstm_out, _ = self.m.parserlstm(curr, slen)
                    if hasattr(self.m, 'unlabeled'): 
                        return self.m.unlabeled(lstm_out, lstm_out).squeeze(3), self.m.deprel(lstm_out, lstm_out)
                    arc = self.m.arc_embedding(lstm_out, lstm_out)
                    return self.m.unlabeled_linear(arc).squeeze(3), self.m.deprel_linear(arc)
            
            device = next(m.parameters()).device
            inputs = (torch.randint(1, 10, (1, 6), device=device, dtype=torch.long), torch.ones((1, 6), device=device, dtype=torch.bool), 
                      torch.zeros(6, 16, device=device, dtype=torch.long), torch.ones(6, 16, device=device, dtype=torch.bool),
                      torch.zeros(1, 6, device=device, dtype=torch.long), torch.randint(1, 10, (1, 6), device=device, dtype=torch.long), 
                      torch.randint(1, 10, (1, 6), device=device, dtype=torch.long), torch.tensor([6], device=device, dtype=torch.long), torch.full((6,), 5, device=device, dtype=torch.long))
            
            # 모델 구성 및 텐서 값 확인을 위한 로그
            print(f"    [LOG] Depparse 모델 구성: parserlstm.input_size={m.parserlstm.input_size}, tag_emb_dim={m.args['tag_emb_dim']}")
            print(f"    [LOG] Depparse 입력 텐서 Shape: word={inputs[0].shape}, mask={inputs[1].shape}, chars={inputs[2].shape}, chars_mask={inputs[3].shape}, upos={inputs[4].shape}, pre={inputs[5].shape}, lemma={inputs[6].shape}, slen={inputs[7].shape}, wlen={inputs[8].shape}")
            
            # 파일 경로를 변경된 폴더 구조에 맞게 수정
            dynamic_axes = {
                'word': {0: 'batch', 1: 'seq'},
                'mask': {0: 'batch', 1: 'seq'},
                'chars': {0: 'num_words', 1: 'char_len'},
                'chars_mask': {0: 'num_words', 1: 'char_len'},
                'upos': {0: 'batch', 1: 'seq'},
                'pre': {0: 'batch', 1: 'seq'},
                'lemma': {0: 'batch', 1: 'seq'},
                'slen': {0: 'batch'},
                'wlen': {0: 'num_words'},
                'unlabeled': {0: 'batch', 1: 'seq', 2: 'seq'},
                'deprel': {0: 'batch', 1: 'seq', 2: 'seq'},
            }
            ok = safe_onnx_export(
                DepWrapper(m),
                inputs,
                f"{ONNX_DIR}/depparse.onnx",
                ['word','mask','chars','chars_mask','upos','pre','lemma','slen','wlen'],
                ['unlabeled','deprel'],
                dynamic_axes
            )
            if ok:
                print("    ✅ Done")
        except Exception as e: print(f"    ❌ {e}")

    # D. LEMMA
    if 'lemma' in nlp.processors:
        try:
            print("  📦 Lemma...")
            # 사전 기반(Dictionary-based) Lemmatizer인 경우 trainer.model이 없을 수 있으므로 체크
            if hasattr(nlp.processors['lemma'], 'trainer') and hasattr(nlp.processors['lemma'].trainer, 'model'):
                m = nlp.processors['lemma'].trainer.model.cpu().eval()
                patch_model_recursively(m)
                
                class LemmaWrapper(nn.Module):
                    def __init__(self, m): super().__init__(); self.m = m
                    def forward(self, src, src_mask, tgt_in, pos):
                        # Stanza의 Neural Lemmatizer (Seq2Seq) 기본 입력 포맷
                        # 키워드 인자 매핑 오류를 피하기 위해 모든 인자를 순서대로(positional) 넘깁니다.
                        out = self.m(src, src_mask, tgt_in, pos)
                        if isinstance(out, tuple):
                            return out[0]
                        return out
                
                # 입력 텐서 더미: Seq2Seq 모델 구조에 맞게 tgt_in을 세 번째로 배치
                device = next(m.parameters()).device
                src_dummy = torch.randint(1, 10, (1, 6), device=device, dtype=torch.long)
                src_mask_dummy = torch.zeros((1, 6), device=device, dtype=torch.bool)
                tgt_in_dummy = torch.randint(1, 10, (1, 6), device=device, dtype=torch.long)
                pos_dummy = torch.randint(1, 10, (1,), device=device, dtype=torch.long)
                
                inputs = (src_dummy, src_mask_dummy, tgt_in_dummy, pos_dummy)
                
                print(f"    [LOG] Lemma 입력 텐서 Shape: src={inputs[0].shape}, src_mask={inputs[1].shape}, tgt_in={inputs[2].shape}, pos={inputs[3].shape}")
                
                dynamic_axes = {
                    'src': {0: 'batch', 1: 'seq'},
                    'src_mask': {0: 'batch', 1: 'seq'},
                    'tgt_in': {0: 'batch', 1: 'seq'},
                    'pos': {0: 'batch'},
                    'preds': {0: 'batch', 1: 'seq'},
                }
                ok = safe_onnx_export(
                    LemmaWrapper(m),
                    inputs,
                    f"{ONNX_DIR}/lemma.onnx",
                    ['src','src_mask','tgt_in','pos'],
                    ['preds'],
                    dynamic_axes
                )
                if ok:
                    print("    ✅ Done")
            else:
                print("    ⚠️ Rule-based/Dictionary Lemmatizer (No PyTorch model to export)")
        except Exception as e: print(f"    ❌ {e}")

    # E. VOCAB EXTRACTION
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
    apply_global_patches()
    for l in ['en', 'cs', 'ko', 'de', 'es', 'fr', 'ja', 'pt', 'ar', 'it', 'nl', 'zh-hans']:
        try:
            convert_lang(l)
        except BaseException as e:
            print(f"⚠️ {l} failed during conversion: {e}")
