import os
from model_manager import model_manager
from processor import process_pii, extract_pii_with_llm
from utils import get_best_language
import stanza

def test_decomposition():
    print("=== Text Decomposition & Deep Detection Test ===")
    
    # 1. Load models
    model_manager.load_llm()
    lang = "ko"
    model_manager.download_stanza_model(lang)
    nlp = model_manager.get_stanza_pipeline(lang)
    
    if not nlp:
        print("Failed to load Stanza for decomposition test.")
        return

    # Long/Complex text for testing
    text = """안녕하세요. 저는 홍길동입니다.
서울시 강남구 테헤란로 123번지에 거주하고 있습니다.
제 개인 이메일은 gildong.hong@gmail.com 이고, 업무용은 work.hgd@company.co.kr 입니다.
문의사항은 010-9876-5432로 연락 부탁드립니다. 감사합니다."""

    print(f"\n[Original Text Length]: {len(text)} chars")
    
    # --- Strategy: Decomposition via Stanza ---
    doc = nlp(text)
    
    print("\n--- [Decomposition Result] ---")
    all_units = []
    
    for i, sent in enumerate(doc.sentences):
        print(f"\n[Unit {i+1} (Sentence)]: {sent.text}")
        
        # Further split sentence into chunks/phrases (using dependencies or just words)
        words = [w.text for w in sent.words]
        print(f"  └─ Tokens: {words}")
        
        # Test detection on this specific sentence
        found = extract_pii_with_llm(sent.text, lang)
        print(f"  └─ LLM Found in this unit: {found}")
        all_units.append(found)

    print("\n--- Final Aggregation ---")
    # Flatten and deduplicate
    final_pii = []
    for unit in all_units:
        final_pii.extend(unit)
    
    print(f"Total Unique PII found via decomposition: {len(set(str(p) for p in final_pii))}")

if __name__ == "__main__":
    test_decomposition()
