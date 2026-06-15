import re
import json
import torch
from config import PII_CATEGORIES
from model_manager import model_manager
from utils import get_best_language

# 정규표현식 패턴
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{3,4}[- ]?\d{4,6}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def extract_pii_with_llm(text, lang_code):
    """IBM Granite-4.0-350M 특화 프롬프트를 이용한 PII 추출"""
    if model_manager.model is None:
        return []

    # 350M 모델의 성능을 끌어올리기 위한 Few-shot 예시 추가
    few_shot = ""
    if lang_code == "ko":
        few_shot = """
Text: 홍길동은 서울시 강남구에 삽니다. 연락처는 010-1234-5678입니다.
JSON: [{"word": "홍길동", "type": "NAME"}, {"word": "서울시 강남구", "type": "ADDRESS"}, {"word": "010-1234-5678", "type": "PHONE"}]
"""
    else:
        few_shot = """
Text: John Doe lives in New York. Call him at +1-202-555-0147.
JSON: [{"word": "John Doe", "type": "NAME"}, {"word": "New York", "type": "ADDRESS"}, {"word": "+1-202-555-0147", "type": "PHONE"}]
"""

    prompt = f"""<|start_of_role|>system<|end_of_role|>You are an expert PII detector. Identify: {', '.join(PII_CATEGORIES)}.
Return ONLY a JSON array of objects with "word" and "type".
{few_shot}<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Language: {lang_code}
Text: {text}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
JSON:"""
    
    inputs = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.model.device)
    with torch.no_grad():
        outputs = model_manager.model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False,
            pad_token_id=model_manager.tokenizer.eos_token_id
        )
    
    response = model_manager.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    try:
        json_str = response.strip()
        match = re.search(r'\[.*\]', json_str, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data if isinstance(data, list) else []
        return []
    except:
        return []

def process_pii(text, mask_char="*"):
    if not text:
        return "입력 텍스트가 비어 있습니다.", "N/A", [], "입력 대기 중"

    # 1. 언어 감지 및 모델 체크
    main_lang, sub_langs = get_best_language(text)
    download_msg = ""
    if not model_manager.is_stanza_installed(main_lang):
        download_msg = f"🔄 {main_lang} 모델 자동 다운로드 중..."
        success, msg = model_manager.download_stanza_model(main_lang)
        download_msg = f"✅ {main_lang} 다운로드 완료!" if success else f"❌ {main_lang} 실패: {msg}"
    else:
        download_msg = f"✨ {main_lang} 모델 준비 완료."

    nlp = model_manager.get_stanza_pipeline(main_lang)
    
    # 2. 텍스트 쪼개기 전략 (Stanza Sentence Splitting)
    print(f"\n[DEBUG] Stanza Decomposition for language: {main_lang}")
    units = []
    if nlp:
        doc = nlp(text)
        units = [sent.text for sent in doc.sentences]
        print(f"[DEBUG] Stanza split text into {len(units)} sentences.")
    else:
        # Fallback to lines
        units = [line.strip() for line in text.split('\n') if line.strip()]
        print(f"[DEBUG] Stanza not available. Split into {len(units)} lines via fallback.")

    results = []
    pos_info = f"Main Language: {main_lang}\n"
    
    # 3. 쪼개진 단위별로 정밀 탐지 수행
    for i, unit in enumerate(units):
        print(f"  > Processing Unit {i+1}/{len(units)}: {unit[:30]}...")
        # Regex (Phone, Email)
        for m in PHONE_REGEX.finditer(unit): results.append({"word": m.group(), "type": "PHONE"})
        for m in EMAIL_REGEX.finditer(unit): results.append({"word": m.group(), "type": "EMAIL"})
        
        # LLM 탐지 (Granite 4.0 Specific)
        llm_found = extract_pii_with_llm(unit, main_lang)
        results.extend(llm_found)
        
        # Stanza NER (있을 경우만)
        if nlp and hasattr(doc, 'entities'):
            for ent in doc.entities:
                if ent.text in unit:
                    results.append({"word": ent.text, "type": ent.type})

    # POS 정보 샘플링
    if nlp and doc.sentences:
        pos_info += " / ".join([f"{w.text}({w.upos})" for w in doc.sentences[0].words[:15]]) + "...\n"

    # 4. 결과 정제 및 마스킹
    unique_results = []
    seen_words = set()
    for res in sorted(results, key=lambda x: len(x.get('word','')), reverse=True):
        word = res.get('word', '')
        if word and word not in seen_words:
            unique_results.append(res)
            seen_words.add(word)

    masked_text = text
    for res in unique_results:
        word = res['word']
        if word in masked_text:
            if mask_char == "[TAG]":
                masked_text = masked_text.replace(word, f"[{res['type']}]")
            else:
                masked_text = masked_text.replace(word, mask_char * len(word))
            
    return masked_text, pos_info, [(r['word'], r['type']) for r in unique_results], download_msg

    # 4. 결과 정제 및 마스킹
    unique_results = []
    seen_words = set()
    for res in sorted(results, key=lambda x: len(x.get('word','')), reverse=True):
        word = res.get('word', '')
        if word and word not in seen_words:
            unique_results.append(res)
            seen_words.add(word)

    masked_text = text
    for res in unique_results:
        word = res['word']
        if word in masked_text:
            if mask_char == "[TAG]":
                masked_text = masked_text.replace(word, f"[{res['type']}]")
            else:
                masked_text = masked_text.replace(word, mask_char * len(word))
            
    return masked_text, pos_info, [(r['word'], r['type']) for r in unique_results], download_msg
