import torch
import stanza
import os
import traceback
import logging

logging.basicConfig(filename='pos_export.log', level=logging.INFO)

def export_pos_standard(lang='ko'):
    output_dir = './onnx_models_pos_test'
    os.makedirs(output_dir, exist_ok=True)
    
    nlp = stanza.Pipeline(lang, processors='tokenize,pos')
    model = nlp.processors['pos'].trainer.model
    model.cpu()
    model.eval()
    
    batch_size, seq_len = 1, 10
    # Create input tuple matching signature exactly
    inputs = (
        torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long), # word
        torch.ones((batch_size, seq_len), dtype=torch.bool), # word_mask
        torch.randint(0, 100, (batch_size, seq_len, 5), dtype=torch.long), # wordchars
        torch.ones((batch_size, seq_len, 5), dtype=torch.bool), # wordchars_mask
        None, None, None, # upos, xpos, ufeats
        torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long), # pretrained
        torch.zeros((batch_size, seq_len), dtype=torch.long), # word_orig_idx
        torch.tensor([seq_len], dtype=torch.long), # sentlens
        torch.ones((batch_size,), dtype=torch.long), # wordlens
        "dummy" # text
    )
    
    input_names = ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens', 'text']
    
    onnx_path = os.path.join(output_dir, f"stanza_{lang}_pos.onnx")
    
    try:
        torch.onnx.export(
            model, inputs, onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=input_names,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        )
        print(f"✅ POS 변환 성공: {onnx_path}")
    except Exception as e:
        print(f"❌ POS 변환 실패: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    export_pos_standard('ko')
