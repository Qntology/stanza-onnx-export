import torch
import torch.nn as nn
import stanza
import os
import traceback
import logging

logging.basicConfig(filename='patch_biaffine.log', level=logging.INFO)

# BiaffineScorer 패치
class PatchedBiaffineScorer(nn.Module):
    def __init__(self, original_scorer):
        super().__init__()
        self.W_bilin = original_scorer.W_bilin
        
    def forward(self, input1, input2):
        # 1. input1/2에 1 패딩 (Biaffine의 bias 처리)
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], -1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], -1)
        
        # 2. W_bilin.weight는 [1, output_size, input1+1, input2+1] 형태일 가능성이 높음
        weight = self.W_bilin.weight.squeeze(0)
        
        # 3. einsum으로 행렬 곱셈 수행
        # input1: [batch, seq1, dim1], weight: [output, dim1, dim2], input2: [batch, seq2, dim2]
        # output: [batch, seq1, seq2, output]
        output = torch.einsum('bxi,oij,byj->bxoy', input1, weight, input2)
        return output.permute(0, 1, 3, 2)

class POSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 패치 적용
        if hasattr(model, 'biaffine_scorer'):
            model.biaffine_scorer = PatchedBiaffineScorer(model.biaffine_scorer)
            
    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens):
        return self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, "dummy")

def export_pos(model, onnx_path):
    model.cpu()
    model.eval()
    wrapper = POSWrapper(model)
    
    batch_size, seq_len = 1, 10
    inputs = (
        torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long),
        torch.ones((batch_size, seq_len), dtype=torch.bool),
        torch.randint(0, 100, (batch_size, seq_len, 5), dtype=torch.long),
        torch.ones((batch_size, seq_len), dtype=torch.bool),
        torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long),
        torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long),
        torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long),
        torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long),
        torch.zeros((batch_size, seq_len), dtype=torch.long),
        torch.tensor([seq_len], dtype=torch.long),
        torch.ones((batch_size,), dtype=torch.long)
    )
    
    input_names = ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens']
    
    torch.onnx.export(
        wrapper, inputs, onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=input_names,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

def run_patch_convert(lang='ko'):
    output_dir = './onnx_models_patched'
    os.makedirs(output_dir, exist_ok=True)
    
    nlp = stanza.Pipeline(lang, processors='tokenize,pos')
    
    for proc_name, processor in nlp.processors.items():
        if proc_name == 'pos':
            onnx_path = os.path.join(output_dir, f"stanza_{lang}_pos_patched.onnx")
            print(f"\n--- 변환 시도: {proc_name} (Biaffine Patch) ---")
            try:
                export_pos(processor.trainer.model, onnx_path)
                print(f"✅ {proc_name} 성공")
            except Exception as e:
                print(f"❌ {proc_name} 실패: {e}")
                logging.error(traceback.format_exc())

if __name__ == "__main__":
    run_patch_convert('ko')
