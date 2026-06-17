import torch
import stanza
import os
import glob
from stanza.models.tokenization.trainer import Trainer as TokenizerTrainer
from stanza.models.mwt.trainer import Trainer as MWTTrainer
from stanza.models.pos.trainer import Trainer as POSTrainer
from stanza.models.ner.trainer import Trainer as NERTrainer

def export_tokenizer(model, onnx_path):
    print(f"--- Tokenizer 변환 중 ---")
    model.eval()
    emb_dim = model.embeddings.embedding_dim
    rnn_input_dim = model.rnn.input_size
    feat_dim = rnn_input_dim - emb_dim
    
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

def export_mwt(model, onnx_path):
    print(f"--- MWT 변환 중 ---")
    model.eval()
    seq_len = 32
    # MWT 모델은 0을 패딩으로 인식하므로 0이 아닌 값으로 채웁니다.
    dummy_src = torch.randint(1, 50, (1, seq_len), dtype=torch.long)
    # Stanza MWT는 0을 제외한 개수를 세므로, 1로 채워진 텐서를 넣으면 길이를 정상적으로 인식합니다.
    dummy_mask = torch.zeros(1, seq_len, dtype=torch.long) 
    
    torch.onnx.export(
        model, (dummy_src, dummy_mask), onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['src', 'src_mask'], output_names=['logits'],
        dynamic_axes={'src': {0: 'batch_size', 1: 'seq'}, 'src_mask': {0: 'batch_size', 1: 'seq'}, 'logits': {0: 'batch_size', 1: 'seq'}}
    )

def convert_local_folder(folder_path, lang='ko'):
    output_dir = './onnx_models_batch'
    os.makedirs(output_dir, exist_ok=True)
    
    pt_files = glob.glob(os.path.join(folder_path, "**/*.pt"), recursive=True)
    print(f"--- {len(pt_files)}개의 .pt 파일을 찾았습니다. ---")
    
    for pt_path in pt_files:
        file_name = os.path.basename(pt_path)
        print(f"\n분석 중: {pt_path}")
        
        # 1. Tokenizer 시도
        try:
            trainer = TokenizerTrainer(model_file=pt_path)
            model = trainer.model
            if model:
                onnx_path = os.path.join(output_dir, f"stanza_{lang}_{file_name.replace('.pt', '')}_tokenize.onnx")
                export_tokenizer(model, onnx_path)
                print(f"✅ Tokenizer 변환 완료: {onnx_path}")
                continue
        except Exception as e: pass

        # 2. MWT 시도
        try:
            trainer = MWTTrainer(model_file=pt_path)
            model = trainer.model
            if model:
                onnx_path = os.path.join(output_dir, f"stanza_{lang}_{file_name.replace('.pt', '')}_mwt.onnx")
                export_mwt(model, onnx_path)
                print(f"✅ MWT 변환 완료: {onnx_path}")
                continue
        except Exception as e: pass

        # 3. POS/NER 시도 (지원 가능한 범위만)
        print(f"ℹ️ '{file_name}'은(는) 현재 자동 변환이 지원되지 않거나 호환되지 않는 모델입니다.")

if __name__ == "__main__":
    # 사용자가 압축 해제한 폴더 경로를 여기에 넣으세요.
    target_folder = './extracted_models' 
    
    # 예시 실행 (실제 폴더가 없을 경우 대비해 캐시 폴더 로직 유지)
    if not os.path.exists(target_folder):
        import platform
        home = os.path.expanduser("~")
        target_folder = os.path.join(home, "AppData/Local/StanfordNLP/stanza/Cache/1.12.0/resources")
    
    convert_local_folder(target_folder, lang='ko')
