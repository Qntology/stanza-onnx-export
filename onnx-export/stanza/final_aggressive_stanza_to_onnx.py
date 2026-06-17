import torch
import torch.nn as nn
import stanza
import os
import traceback
import logging

# 로그 설정
logging.basicConfig(filename='final_aggressive_conversion.log', level=logging.INFO)

class POSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens):
        # upos, xpos, ufeats는 모델 내부에서 로직에 따라 사용되므로 None으로 처리 가능하면 처리
        return self.model(word, word_mask, wordchars, wordchars_mask, None, None, None, pretrained, word_orig_idx, sentlens, wordlens, "dummy")

class DepparseWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens):
        # pos, lemma, deprel, head는 None으로 처리
        return self.model(word, word_mask, wordchars, wordchars_mask, None, None, None, None, word_orig_idx, sentlens, wordlens, "dummy")

def export_with_wrapper(model, wrapper_class, proc_name, onnx_path):
    model.cpu()
    model.eval()
    wrapper = wrapper_class(model)
    
    # 1. Inputs (word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens)
    batch_size, seq_len = 1, 10
    inputs = (
        torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long),
        torch.ones((batch_size, seq_len), dtype=torch.bool),
        torch.randint(0, 100, (batch_size, seq_len, 5), dtype=torch.long),
        torch.ones((batch_size, seq_len, 5), dtype=torch.bool),
        torch.randn(batch_size, seq_len, 125, dtype=torch.float),
        torch.zeros((batch_size, seq_len), dtype=torch.long),
        torch.tensor([seq_len], dtype=torch.long),
        torch.ones((batch_size,), dtype=torch.long)
    )
    
    input_names = ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens']
    
    torch.onnx.export(
        wrapper, inputs, onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=input_names,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

def aggressive_final_convert(lang='ko'):
    output_dir = './onnx_models_final_aggressive'
    os.makedirs(output_dir, exist_ok=True)
    
    nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
    
    for proc_name, processor in nlp.processors.items():
        if not hasattr(processor, 'trainer') or not processor.trainer.model:
            continue
        
        onnx_path = os.path.join(output_dir, f"stanza_{lang}_{proc_name}.onnx")
        print(f"\n--- 변환 시도: {proc_name} ---")
        
        try:
            if proc_name == 'pos':
                export_with_wrapper(processor.trainer.model, POSWrapper, proc_name, onnx_path)
                print(f"✅ {proc_name} 성공")
            elif proc_name == 'depparse':
                export_with_wrapper(processor.trainer.model, DepparseWrapper, proc_name, onnx_path)
                print(f"✅ {proc_name} 성공")
        except Exception as e:
            err_msg = f"{proc_name} 실패: {str(e)}\n{traceback.format_exc()}"
            logging.error(err_msg)
            print(f"❌ {proc_name} 실패 (로그 확인: final_aggressive_conversion.log)")

if __name__ == "__main__":
    aggressive_final_convert('ko')
