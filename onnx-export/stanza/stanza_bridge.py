import stanza
import sys
import os
from pii.processor import process_pii, model_manager

# PII 모듈 및 utils를 찾을 수 있도록 경로 추가
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pii')))

# 직접 Stanza 로드
nlp = stanza.Pipeline('ko', processors='tokenize,pos,lemma,depparse')

# Mock manager 업데이트
model_manager.model = None 

def run_pii_masking(text):
    # Stanza 분석
    # PII processor 내부에서 nlp를 사용해야 하는데, 이를 위해 
    # model_manager의 메서드를 수정하거나 직접 nlp 객체를 주입해야 함.
    # 일단은 process_pii(text)가 내부적으로 model_manager.load_stanza를 쓰므로
    # 이 호출을 가로채서 이미 로드된 nlp를 반환하게 해야 함.
    
    # 임시 우회: model_manager가 nlp를 리턴하도록 패치
    model_manager.load_stanza = lambda lang, processors: nlp
    
    # 로직 수행
    masked_text, _, _, _ = process_pii(text)
    return masked_text
