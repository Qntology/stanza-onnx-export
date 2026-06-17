import torch
import torch.nn as nn
import stanza
import os
import traceback
import logging

logging.basicConfig(filename='jit_conversion.log', level=logging.INFO)

class POSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens):
        return self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, "dummy")

class DepparseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens):
        return self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens, "dummy")

def export_complex_model_jit(model, wrapper_class, proc_name, onnx_path, inputs, input_names):
    model.cpu()
    model.eval()
    wrapper = wrapper_class(model)
    
    # JIT Trace
    traced_model = torch.jit.trace(wrapper, inputs)
    
    torch.onnx.export(
        traced_model, inputs, onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=input_names,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

def get_complex_dummy_inputs(proc_name):
    batch_size, seq_len = 1, 10
    # 공통 입력들
    word = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long)
    word_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    wordchars = torch.randint(0, 100, (batch_size, seq_len, 5), dtype=torch.long)
    wordchars_mask = torch.ones((batch_size, seq_len, 5), dtype=torch.bool)
    
    # 텐서 기반 더미 입력 (None 대신)
    dummy_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    if proc_name == 'pos':
        return (word, word_mask, wordchars, wordchars_mask, dummy_tensor, dummy_tensor, dummy_tensor,
                torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long), # Pretrained
                torch.zeros((batch_size, seq_len), dtype=torch.long),
                torch.tensor([seq_len], dtype=torch.long),
                torch.ones((batch_size,), dtype=torch.long)), ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens']
                
    elif proc_name == 'depparse':
        return (word, word_mask, wordchars, wordchars_mask, dummy_tensor, dummy_tensor, dummy_tensor,
                torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long), # Pretrained
                dummy_tensor, dummy_tensor, dummy_tensor,
                torch.zeros((batch_size, seq_len), dtype=torch.long),
                torch.tensor([seq_len], dtype=torch.long),
                torch.ones((batch_size,), dtype=torch.long)), ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'lemma', 'head', 'deprel', 'word_orig_idx', 'sentlens', 'wordlens']
    return None, None

def aggressive_jit_convert(lang='ko'):
    output_dir = './onnx_models_jit'
    os.makedirs(output_dir, exist_ok=True)
    
    nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
    
    for proc_name, processor in nlp.processors.items():
        if not hasattr(processor, 'trainer') or not processor.trainer.model:
            continue
        
        onnx_path = os.path.join(output_dir, f"stanza_{lang}_{proc_name}.onnx")
        print(f"\n--- 변환 시도: {proc_name} ---")
        
        try:
            model = processor.trainer.model
            inputs, input_names = get_complex_dummy_inputs(proc_name)
            
            if inputs:
                wrapper = POSWrapper if proc_name == 'pos' else DepparseWrapper
                export_complex_model_jit(model, wrapper, proc_name, onnx_path, inputs, input_names)
                print(f"✅ {proc_name} 성공")
            else:
                print(f"ℹ️ {proc_name}는 변환 로직이 구현되지 않았습니다.")
        except Exception as e:
            err_msg = f"{proc_name} 실패: {str(e)}\n{traceback.format_exc()}"
            logging.error(err_msg)
            print(f"❌ {proc_name} 실패 (로그 확인: jit_conversion.log)")

if __name__ == "__main__":
    aggressive_jit_convert('ko')
