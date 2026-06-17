import torch
import stanza
import os
import zipfile
import requests
import numpy as np
import onnxruntime as ort
from stanza.models.common.trainer import Trainer

def convert_to_onnx(model, name, lang, output_dir):
    model.cpu()
    model.eval()
    onnx_path = os.path.join(output_dir, f"stanza_{lang}_{name}.onnx")
    
    try:
        # 1. Tokenizer 아키텍처 (embeddings, rnn)
        if hasattr(model, 'embeddings') and hasattr(model, 'rnn'):
            emb_dim = model.embeddings.embedding_dim
            rnn_input_dim = model.rnn.input_size
            feat_dim = rnn_input_dim - emb_dim
            
            print(f"--- {name} (Tokenizer) 변환 중 ---")
            seq_len = 128
            dummy_input = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
            dummy_feats = torch.zeros(1, seq_len, feat_dim, dtype=torch.float)
            dummy_lengths = torch.tensor([seq_len], dtype=torch.long)
            
            torch.onnx.export(
                model, (dummy_input, dummy_feats, dummy_lengths), onnx_path,
                export_params=True, opset_version=14, do_constant_folding=True,
                input_names=['input', 'feats', 'lengths'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size', 1: 'seq'}, 'feats': {0: 'batch_size', 1: 'seq'}, 'lengths': {0: 'batch_size'}, 'output': {0: 'batch_size', 1: 'seq'}}
            )
            print(f"✅ {name} 변환 완료: {onnx_path}")
            return True

        # 2. MWT 아키텍처 (embedding, encoder)
        elif hasattr(model, 'embedding') and hasattr(model, 'encoder'):
            print(f"--- {name} (MWT/Classifier) 변환 중 ---")
            seq_len = 32
            dummy_src = torch.randint(1, 50, (1, seq_len), dtype=torch.long)
            # MWT 모델은 패딩(0)을 제외한 길이를 계산하므로 모두 1로 채웁니다.
            dummy_mask = torch.ones(1, seq_len, dtype=torch.long) 
            
            torch.onnx.export(
                model, (dummy_src, dummy_mask), onnx_path,
                export_params=True, opset_version=14, do_constant_folding=True,
                input_names=['src', 'src_mask'], output_names=['logits'],
                dynamic_axes={'src': {0: 'batch_size', 1: 'seq'}, 'src_mask': {0: 'batch_size', 1: 'seq'}, 'logits': {0: 'batch_size', 1: 'seq'}}
            )
            print(f"✅ {name} 변환 완료: {onnx_path}")
            return True
        
        else:
            print(f"ℹ️ {name} 모델은 현재 지원하지 않는 아키텍처입니다.")
    except Exception as e:
        print(f"❌ {name} 변환 중 에러 발생: {e}")
    
    return False

def batch_convert(lang):
    print(f"\n--- {lang} 모델 처리 시작 ---")
    
    # 해당 언어의 기본 프로세서 확인
    try:
        # Pipeline을 생성할 때 알아서 다운로드하게 둡니다.
        # (한국어처럼 mwt가 없는 경우를 위해 tokenize,pos만 먼저 시도)
        processors = 'tokenize,pos'
        if lang in ['en', 'fr', 'de', 'es']: # mwt가 확실히 있는 언어들
            processors += ',mwt'
            
        nlp = stanza.Pipeline(lang, processors=processors)
    except Exception as e:
        print(f"ℹ️ 파이프라인 생성 중 경고 (일부 프로세서 무시 가능): {e}")
        nlp = stanza.Pipeline(lang, processors='tokenize')
    
    output_dir = './onnx_models'
    os.makedirs(output_dir, exist_ok=True)
    
    for proc_name, processor in nlp.processors.items():
        if hasattr(processor, 'trainer') and processor.trainer.model:
            convert_to_onnx(processor.trainer.model, proc_name, lang, output_dir)

if __name__ == "__main__":
    # 사용 예: batch_convert('ko') 또는 batch_convert('en')
    batch_convert('ko')
    batch_convert('en')
