import torch
import stanza
import os

def extract_pos_weights(lang='ko'):
    print(f"--- {lang} 모델 로드 및 가중치 추출 ---")
    nlp = stanza.Pipeline(lang, processors='tokenize,pos')
    model = nlp.processors['pos'].trainer.model
    
    # 모델의 모든 가중치를 state_dict로 추출
    state_dict = model.state_dict()
    
    # torch 포맷으로 저장 (Rust의 tch가 읽을 수 있음)
    output_path = f"stanza_{lang}_pos_weights.pt"
    torch.save(state_dict, output_path)
    print(f"✅ 가중치 추출 완료: {output_path}")

if __name__ == "__main__":
    extract_pos_weights('ko')
