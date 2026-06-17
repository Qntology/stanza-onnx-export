import torch
import stanza
import os
import glob
from stanza.models.common.trainer import Trainer
from stanza.models.tokenization.model import Tokenizer
from stanza.models.mwt.character_classifier import CharacterClassifier

def get_model_type(model):
    if isinstance(model, Tokenizer):
        return "tokenize"
    if isinstance(model, CharacterClassifier):
        return "mwt"
    # POS/NER 등은 로직이 복잡하여 기본적으로 아키텍처 패턴으로 파악
    if hasattr(model, 'embeddings') and hasattr(model, 'rnn'):
        return "tokenize" # Tokenizer-like
    if hasattr(model, 'embedding') and hasattr(model, 'encoder'):
        return "mwt" # MWT-like
    return "unknown"

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
    dummy_src = torch.randint(1, 50, (1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long) 
    
    torch.onnx.export(
        model, (dummy_src, dummy_mask), onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['src', 'src_mask'], output_names=['logits'],
        dynamic_axes={'src': {0: 'batch_size', 1: 'seq'}, 'src_mask': {0: 'batch_size', 1: 'seq'}, 'logits': {0: 'batch_size', 1: 'seq'}}
    )

from stanza.models.tokenization.trainer import Trainer as TokenizerTrainer
from stanza.models.mwt.trainer import Trainer as MWTTrainer

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
        except:
            pass

        # 2. MWT 시도
        try:
            trainer = MWTTrainer(model_file=pt_path)
            model = trainer.model
            if model:
                onnx_path = os.path.join(output_dir, f"stanza_{lang}_{file_name.replace('.pt', '')}_mwt.onnx")
                export_mwt(model, onnx_path)
                print(f"✅ MWT 변환 완료: {onnx_path}")
                continue
        except:
            pass

        print(f"ℹ️ '{file_name}'은(는) 호환되는 Tokenizer 또는 MWT 모델이 아닙니다.")

if __name__ == "__main__":
    # 사용자가 압축 해제한 폴더 경로를 입력하세요.
    # 예: batch_convert_local('./my_extracted_models', lang='ko')
    
    # 테스트를 위해 Stanza 캐시 폴더를 예시로 사용할 수도 있습니다.
    # (이미 다운로드된 파일이 있다면 그 경로를 넣어보세요)
    import platform
    home = os.path.expanduser("~")
    stanza_cache = os.path.join(home, "stanza_resources" if platform.system() != "Windows" else "AppData/Local/StanfordNLP/stanza/Cache/1.12.0/resources")
    
    # 실제 사용 시에는 압축 해제한 경로를 넣으시면 됩니다.
    convert_local_folder(stanza_cache, lang='ko')
