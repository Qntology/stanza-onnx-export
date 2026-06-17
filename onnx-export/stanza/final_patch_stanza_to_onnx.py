import torch
import torch.nn as nn
import stanza
import os
import traceback
import logging

logging.basicConfig(filename='final_patch_conversion.log', level=logging.INFO)

def patch_model_forward(model):
    """Patch the model's forward method to ignore the 'text' argument."""
    original_forward = model.forward
    def new_forward(*args, **kwargs):
        # 'text' is the last argument, ignore it.
        return original_forward(*args[:-1])
    model.forward = new_forward
    return model

def patch_model_dependencies(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and module.weight is None:
            module.weight = nn.Parameter(torch.randn(module.num_embeddings, module.embedding_dim))
    return model

def export_complex_model(model, inputs, onnx_path, input_names):
    model.cpu()
    model.eval()
    model = patch_model_dependencies(model)
    model = patch_model_forward(model)
    
    torch.onnx.export(
        model, inputs, onnx_path,
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
    output_dir = './onnx_models_patched'
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
                export_complex_model(model, inputs, onnx_path, input_names)
                print(f"✅ {proc_name} 성공")
            else:
                print(f"ℹ️ {proc_name}는 변환 로직이 구현되지 않았습니다.")
        except Exception as e:
            err_msg = f"{proc_name} 실패: {str(e)}\n{traceback.format_exc()}"
            logging.error(err_msg)
            print(f"❌ {proc_name} 실패 (로그 확인: final_patch_conversion.log)")

if __name__ == "__main__":
    aggressive_final_convert('ko')
