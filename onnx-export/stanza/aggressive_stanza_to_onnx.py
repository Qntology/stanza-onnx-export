import torch
import stanza
import os
import logging
import traceback
import torch.nn as nn

# 로그 설정
logging.basicConfig(filename='aggressive_conversion.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def patch_model_dependencies(model):
    """모델 로딩 오류를 방지하기 위해 필요한 의존성(Embedding 등)을 강제로 생성하여 주입."""
    # POS/Depparse 모델에서 자주 발생하는 'NoneType' 에러 방지
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and module.weight is None:
            print(f"--- Patching {name} with dummy weights ---")
            module.weight = nn.Parameter(torch.randn(module.num_embeddings, module.embedding_dim))
    return model

def export_complex_model(model, inputs, onnx_path, input_names):
    model.cpu()
    model.eval()
    model = patch_model_dependencies(model)
    
    torch.onnx.export(
        model, inputs, onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=input_names,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

def get_complex_dummy_inputs(proc_name):
    batch_size, seq_len = 1, 10
    if proc_name == 'pos':
        # Tagger forward: (self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text)
        return (torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long),
                torch.ones((batch_size, seq_len), dtype=torch.bool),
                torch.randint(0, 100, (batch_size, seq_len, 5), dtype=torch.long),
                torch.ones((batch_size, seq_len, 5), dtype=torch.bool),
                None, None, None, 
                torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long), # Pretrained (Long index)
                torch.zeros((batch_size, seq_len), dtype=torch.long),
                torch.tensor([seq_len], dtype=torch.long),
                torch.ones((batch_size,), dtype=torch.long), # Flattened wordlens
                ["dummy"]), ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens', 'text']
    elif proc_name == 'depparse':
        # GraphParser forward: (self, word, word_mask, wordchars, wordchars_mask, pos, lemma, deprel, head, word_orig_idx, sentlens, wordlens, text)
        return (torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long),
                torch.ones((batch_size, seq_len), dtype=torch.bool),
                torch.randint(0, 100, (batch_size, seq_len, 5), dtype=torch.long),
                torch.ones((batch_size, seq_len, 5), dtype=torch.bool),
                None, None, None, None,
                torch.zeros((batch_size, seq_len), dtype=torch.long),
                torch.tensor([seq_len], dtype=torch.long),
                torch.ones((batch_size,), dtype=torch.long), # Flattened wordlens
                "dummy"), ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'pos', 'lemma', 'deprel', 'head', 'word_orig_idx', 'sentlens', 'wordlens', 'text']
    return None, None

def aggressive_convert(lang='ko'):
    output_dir = './onnx_models_aggressive'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 의존성 무시하고 강제 Pipeline 로드
    nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
    
    for proc_name, processor in nlp.processors.items():
        if not hasattr(processor, 'trainer') or not processor.trainer.model:
            continue
            
        print(f"\n--- Aggressive 변환 시도: {proc_name} ---")
        onnx_path = os.path.join(output_dir, f"stanza_{lang}_{proc_name}.onnx")
        
        try:
            model = processor.trainer.model
            inputs, input_names = get_complex_dummy_inputs(proc_name)
            
            if inputs:
                export_complex_model(model, inputs, onnx_path, input_names)
                print(f"✅ {proc_name} 성공")
            else:
                print(f"ℹ️ {proc_name}는 변환 로직이 구현되지 않았습니다.")
                
        except Exception as e:
            err_msg = f"{proc_name} 실패: {str(e)}\n{traceback.format_exc()}"
            logging.error(err_msg)
            print(f"❌ {proc_name} 실패 (로그 확인: aggressive_conversion.log)")

if __name__ == "__main__":
    aggressive_convert('ko')
