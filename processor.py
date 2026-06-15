import re
import json
import torch
from model_manager import model_manager
from utils import get_best_language

# 이메일/전화번호 정규표현식 (최소한의 안전망으로 유지)
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{3,4}[- ]?\d{4}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def extract_specific_pii(text, target_label, description):
    """특정 카테고리의 PII 하나만 LLM에 요청 (350M 모델 최적화)"""
    if model_manager.model is None:
        return []

    # 지시사항을 극도로 단순화하여 오답 방지 (확신이 있을 때만 추출하도록 강조)
    prompt = f"""<|start_of_role|>system<|end_of_role|>Task: Extract ONLY the real '{target_label}' ({description}) from the text.
Rules:
1. Return JSON array: [{{"word": "actual_text", "type": "{target_label}"}}]
2. If the text does NOT contain a real '{target_label}', return [].
3. DO NOT guess. If unsure, return [].
<|start_of_role|>user<|end_of_role|>Text: {text}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
JSON:"""
    
    inputs = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.model.device)
    with torch.no_grad():
        outputs = model_manager.model.generate(
            **inputs, 
            max_new_tokens=64,
            do_sample=False,
            use_cache=False,
            pad_token_id=model_manager.tokenizer.eos_token_id
        )
    
    response = model_manager.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    try:
        match = re.search(r'\[.*\]', response.strip(), re.DOTALL)
        if match:
            items = json.loads(match.group())
            valid_items = []
            for i in items:
                if not isinstance(i, dict): continue
                word = str(i.get('word', '')).strip()
                # 필터링 조건: 
                # 1. 원본 텍스트에 존재해야 함
                # 2. 라벨 이름이나 placeholder가 아니어야 함
                # 3. 길이가 1보다 커야 함
                if (word in text and 
                    word.lower() not in [target_label.lower(), "found_text", "pii content", "extracted text"] and
                    not word.startswith("found_") and
                    len(word) > 1):
                    valid_items.append(i)
            return valid_items
        return []
    except:
        return []

def process_pii(text, mask_char="*", minimize_address=False):
    if not text:
        return "입력 텍스트가 비어 있습니다.", "N/A", [], "입력 대기 중"

    main_lang, _ = get_best_language(text)
    
    # 1. 텍스트 분할 (줄 단위/문장 단위)
    nlp = model_manager.load_stanza(main_lang)
    if nlp:
        doc = nlp(text)
        units = [sent.text for sent in doc.sentences]
        model_manager.unload_stanza()
    else:
        units = [line.strip() for line in text.split('\n') if line.strip()]

    # 2. LLM 로드
    success, msg = model_manager.load_llm()
    if not success:
        return "LLM 로드 실패", "N/A", [], f"Error: {msg}"
    
    results = []
    # 순차적으로 물어볼 카테고리 (더 구체적인 설명 추가)
    target_categories = [
        ("FIRSTNAME", "Individual person's name"),
        ("STATE", "Province, State, or Do name like '경기도', 'California', 'CA'"),
        ("CITY", "City, District, or Si/Gun/Gu name like '서울시', '성남시', 'San Francisco'"),
        ("STREET", "Road name, Street address, or Building number like '판교역로 166', 'Market St 1234'"),
        ("ZIPCODE", "Postal or Zip code like '13487', '94103'")
    ]

    # --- DOUBLE LOOP STRATEGY ---
    for unit in units:
        # A. 정규식 탐지 (전화번호, 이메일은 빠르므로 유지)
        for m in PHONE_REGEX.finditer(unit): results.append({"word": m.group(), "type": "PHONE"})
        for m in EMAIL_REGEX.finditer(unit): results.append({"word": m.group(), "type": "EMAIL"})
        
        # B. 속성별 개별 LLM 탐지 (주소 등 복잡한 것들)
        for label, desc in target_categories:
            found = extract_specific_pii(unit, label, desc)
            results.extend(found)

    # 3. 결과 정제 (중복 및 짧은 단어 제거)
    unique_results = []
    seen_words = set()
    for res in sorted(results, key=lambda x: len(str(x.get('word',''))), reverse=True):
        word = str(res.get('word', '')).strip()
        if word and word not in seen_words and len(word) > 1:
            unique_results.append(res)
            seen_words.add(word)

    # 4. 마스킹 적용
    masked_text = text
    final_entities = []
    
    # 주소 최소화 로직용 타겟
    address_mask_targets = ["STREET", "ZIPCODE", "BUILDINGNUMBER"]
    address_expose_targets = ["CITY", "STATE", "COUNTY"]

    for res in unique_results:
        word, p_type = res['word'], res['type']
        final_entities.append((word, p_type))
        
        if minimize_address and p_type in address_mask_targets:
            if word in masked_text:
                masked_text = masked_text.replace(word, mask_char * len(word))
        elif minimize_address and p_type in address_expose_targets:
            pass # 노출 유지
        else:
            # 기본 마스킹 (이름, 전화번호, 이메일 등)
            if word in masked_text:
                masked_text = masked_text.replace(word, mask_char * len(word))
            
    return masked_text, f"Main Lang: {main_lang}", final_entities, "✨ Detection Complete"
