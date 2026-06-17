import torch
import stanza
import os
import logging
import traceback
from stanza.models.tokenization.trainer import Trainer as TokenizerTrainer
from stanza.models.mwt.trainer import Trainer as MWTTrainer
from stanza.models.pos.trainer import Trainer as POSTrainer
from stanza.models.lemma.trainer import Trainer as LemmaTrainer
from stanza.models.depparse.trainer import Trainer as DepparseTrainer

# 로그 설정
logging.basicConfig(filename='conversion_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def export_model(model, inputs, onnx_path, input_names):
    model.cpu()
    model.eval()
    torch.onnx.export(
        model, inputs, onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=input_names,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

def get_dummy_inputs(model, proc_name):
    batch_size, seq_len = 1, 10
    if proc_name == 'tokenize':
        emb_dim = model.embeddings.embedding_dim
        rnn_input_dim = model.rnn.input_size
        feat_dim = rnn_input_dim - emb_dim
        return (torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long),
                torch.zeros(batch_size, seq_len, feat_dim, dtype=torch.float),
                torch.tensor([seq_len], dtype=torch.long)), ['input', 'feats', 'lengths']
    elif proc_name == 'mwt':
        return (torch.randint(1, 50, (batch_size, seq_len), dtype=torch.long),
                torch.ones(batch_size, seq_len, dtype=torch.long)), ['src', 'src_mask']
    elif proc_name == 'pos':
        # Simple inputs for POS
        return (torch.randint(0, 100, (batch_size, seq_len), dtype=torch.long),
                torch.ones((batch_size, seq_len), dtype=torch.bool),
                torch.randint(0, 100, (batch_size, seq_len, 5), dtype=torch.long),
                torch.ones((batch_size, seq_len, 5), dtype=torch.bool),
                None, None, None, torch.randn(batch_size, seq_len, 100),
                torch.zeros((batch_size, seq_len), dtype=torch.long),
                torch.tensor([seq_len], dtype=torch.long),
                torch.ones((batch_size, seq_len), dtype=torch.long),
                ["dummy"]), ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens', 'text']
    else:
        raise ValueError(f"Unsupported processor: {proc_name}")

def convert_full_pipeline(lang='ko'):
    output_dir = './onnx_models_full'
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- {lang} 모델 파이프라인 로드 ---")
    
    # 지원 가능한 프로세서만 필터링하여 로드
    all_processors = ['tokenize', 'mwt', 'pos', 'lemma', 'depparse']
    supported_processors = []
    
    for p in all_processors:
        try:
            stanza.Pipeline(lang, processors=p, download_method=None)
            supported_processors.append(p)
        except:
            print(f"ℹ️ {p}는 {lang} 언어를 지원하지 않습니다. 건너뜁니다.")
    
    nlp = stanza.Pipeline(lang, processors=','.join(supported_processors))
    
    for proc_name, processor in nlp.processors.items():
        if not hasattr(processor, 'trainer') or not processor.trainer.model:
            continue
        
        onnx_path = os.path.join(output_dir, f"stanza_{lang}_{proc_name}.onnx")
        print(f"--- 변환 시도: {proc_name} ---")
        
        try:
            inputs, input_names = get_dummy_inputs(processor.trainer.model, proc_name)
            export_model(processor.trainer.model, inputs, onnx_path, input_names)
            print(f"✅ {proc_name} 성공")
        except Exception as e:
            err_msg = f"{proc_name} 변환 실패: {str(e)}\n{traceback.format_exc()}"
            logging.error(err_msg)
            print(f"❌ {proc_name} 실패 (로그 확인: conversion_errors.log)")

if __name__ == "__main__":
    convert_full_pipeline('ko')
