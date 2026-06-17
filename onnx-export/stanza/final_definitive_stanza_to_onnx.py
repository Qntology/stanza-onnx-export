import torch
import torch.nn as nn
import stanza
import os
import traceback
import logging

logging.basicConfig(filename='final_definitive_conversion.log', level=logging.INFO)

class POSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens):
        # Force all indices to Long
        return self.model(word.long(), word_mask, wordchars.long(), wordchars_mask, 
                          upos.long() if upos is not None else None, 
                          xpos.long() if xpos is not None else None, 
                          ufeats.long() if ufeats is not None else None, 
                          pretrained.long(), word_orig_idx.long(), sentlens.long(), wordlens.long(), "dummy")

class DepparseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens):
        # Force all indices to Long
        return self.model(word.long(), word_mask, wordchars.long(), wordchars_mask, 
                          upos.long() if upos is not None else None, 
                          xpos.long() if xpos is not None else None, 
                          ufeats.long() if ufeats is not None else None, 
                          pretrained.long(), 
                          lemma.long() if lemma is not None else None, 
                          head.long() if head is not None else None, 
                          deprel.long() if deprel is not None else None, 
                          word_orig_idx.long(), sentlens.long(), wordlens.long(), "dummy")

def export_with_wrapper(model, wrapper_class, proc_name, onnx_path, inputs, input_names):
    model.cpu()
    model.eval()
    wrapper = wrapper_class(model)
    
    torch.onnx.export(
        wrapper, inputs, onnx_path,
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
    
    dummy_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    if proc_name == 'pos':
        # (word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens)
        return (word, word_mask, wordchars, wordchars_mask, dummy_tensor, dummy_tensor, dummy_tensor,
                torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long), 
                torch.zeros((batch_size, seq_len), dtype=torch.long),
                torch.tensor([seq_len], dtype=torch.long),
                torch.ones((batch_size,), dtype=torch.long)), ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens']
                
    elif proc_name == 'depparse':
        # (word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)
        return (word, word_mask, wordchars, wordchars_mask, dummy_tensor, dummy_tensor, dummy_tensor,
                torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long), 
                dummy_tensor, dummy_tensor, dummy_tensor,
                torch.zeros((batch_size, seq_len), dtype=torch.long),
                torch.tensor([seq_len], dtype=torch.long),
                torch.ones((batch_size,), dtype=torch.long)), ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'lemma', 'head', 'deprel', 'word_orig_idx', 'sentlens', 'wordlens']
    return None, None

def aggressive_final_convert(lang='ko'):
    output_dir = './onnx_models_final'
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
                export_with_wrapper(model, wrapper, proc_name, onnx_path, inputs, input_names)
                print(f"✅ {proc_name} 성공")
            else:
                print(f"ℹ️ {proc_name}는 변환 로직이 구현되지 않았습니다.")
        except Exception as e:
            err_msg = f"{proc_name} 실패: {str(e)}\n{traceback.format_exc()}"
            logging.error(err_msg)
            print(f"❌ {proc_name} 실패 (로그 확인: final_defensive_conversion.log)")

if __name__ == "__main__":
    aggressive_final_convert('ko')
