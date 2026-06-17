import torch
import stanza
import os
import numpy as np
import onnxruntime as ort

def convert_tokenizer(lang='ko', output_dir='./onnx_models'):
    print(f"--- {lang} Tokenizer 변환 시작 ---")
    stanza.download(lang, processors='tokenize')
    nlp = stanza.Pipeline(lang, processors='tokenize')
    
    tokenizer_processor = nlp.processors['tokenize']
    model = tokenizer_processor.trainer.model
    model.cpu() # Ensure model is on CPU
    model.eval()

    # Stanza Tokenizer typically expects: x, feats, lengths
    # Based on model inspection: LSTM input(37) - Embedding(32) = feats(5)
    seq_len = 128
    dummy_input = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
    dummy_feats = torch.zeros(1, seq_len, 5, dtype=torch.float)
    dummy_lengths = torch.tensor([seq_len], dtype=torch.long)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    onnx_path = os.path.join(output_dir, f"stanza_{lang}_tokenizer.onnx")
    
    torch.onnx.export(
        model,
        (dummy_input, dummy_feats, dummy_lengths), # x, feats, lengths
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input', 'feats', 'lengths'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'feats': {0: 'batch_size', 1: 'sequence_length'},
            'lengths': {0: 'batch_size'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print(f"✅ Tokenizer ONNX 저장 완료: {onnx_path}")
    return onnx_path

def verify_onnx(onnx_path):
    print(f"--- ONNX 모델 검증 시작: {onnx_path} ---")
    try:
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        feats_name = session.get_inputs()[1].name
        lengths_name = session.get_inputs()[2].name
        
        # Test with random input
        batch_size = 1
        seq_len = 64
        test_input = np.random.randint(0, 100, (batch_size, seq_len)).astype(np.int64)
        test_feats = np.zeros((batch_size, seq_len, 5)).astype(np.float32)
        test_lengths = np.array([seq_len], dtype=np.int64)
        
        outputs = session.run(None, {
            input_name: test_input,
            feats_name: test_feats,
            lengths_name: test_lengths
        })
        
        print("✅ ONNX Runtime 검증 성공!")
        print(f"출력 형태: {outputs[0].shape}")
    except Exception as e:
        print(f"❌ ONNX Runtime 검증 실패: {e}")

if __name__ == "__main__":
    path = convert_tokenizer(lang='ko')
    verify_onnx(path)
