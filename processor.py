import re
import json
import torch
from model_manager import model_manager
from utils import get_best_language

# 이메일/전화번호 정규표현식 (최소한의 안전망으로 유지)
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{3,4}[- ]?\d{4}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def extract_specific_pii(text, target_label, description, keywords=None):
    """특정 카테고리의 PII 하나만 LLM에 요청 (Stanza 힌트 활용)"""
    if model_manager.model is None:
        return []

    # Stanza가 찾은 힌트가 있다면 프롬프트에 추가
    hint_str = f"\nPotential Keywords: {', '.join(keywords)}" if keywords else ""

    # 카테고리별 예시 제공 (한국어 사례 포함)
    examples = {
        "FIRSTNAME": "Text: '제 이름은 김철수입니다.' -> [{'word': '김철수', 'type': 'FIRSTNAME'}]",
        "ADDRESS": "Text: '서울특별시 강남구 테헤란로 123길 45' -> [{'word': '서울특별시 강남구 테헤란로 123길 45', 'type': 'ADDRESS'}]",
        "STATE": "Text: '경기도 성남시에 삽니다.' -> [{'word': '경기도', 'type': 'STATE'}]",
        "CITY": "Text: '서울특별시 강남구' -> [{'word': '서울특별시', 'type': 'CITY'}, {'word': '강남구', 'type': 'CITY'}]",
        "STREET": "Text: '판교역로 166 (삼성동)' -> [{'word': '판교역로 166', 'type': 'STREET'}]",
        "ZIPCODE": "Text: '우편번호는 06655입니다.' -> [{'word': '06655', 'type': 'ZIPCODE'}]"
    }
    example_str = examples.get(target_label, "")

    prompt = f"""<|start_of_role|>system<|end_of_role|>Extract ONLY '{target_label}' ({description}) from the text.
Rules:
1. Return JSON array: [{{"word": "extracted_text", "type": "{target_label}"}}]
2. Only use words present in the original text.{hint_str}
3. Example: {example_str}
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
    
    # 무시할 일반 단어 (Blacklist)
    blacklist = {"PII content", "Hello", "Thank you", "Notice", "Project", "My office", "located at", "문의", "contact", "International"}
    
    try:
        match = re.search(r'\[.*\]', response.strip(), re.DOTALL)
        if match:
            items = json.loads(match.group())
            valid_items = []
            for i in items:
                if not isinstance(i, dict): continue
                word = str(i.get('word', '')).strip()
                if (word in text and 
                    word.lower() not in [target_label.lower(), "found_text", "pii content", "extracted text"] and
                    word not in blacklist and
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
    
    # --- [Step 1] Stanza 분석 (POS 힌트 추출) ---
    nlp = model_manager.load_stanza(main_lang, processors='tokenize,pos')
    units_data = []
    if nlp:
        doc = nlp(text)
        for sent in doc.sentences:
            # PII 후보가 될 만한 고유명사, 명사, 숫자 추출
            keywords = []
            for word in sent.words:
                if word.upos in ['PROPN', 'NOUN', 'NUM'] or word.xpos in ['nq', 'ncn', 'nnc']:
                    keywords.append(word.text)
            units_data.append({
                "text": sent.text,
                "keywords": list(set(keywords))
            })
        model_manager.unload_stanza()
    else:
        units_data = [{"text": line.strip(), "keywords": []} for line in text.split('\n') if line.strip()]

    # --- [Step 2] LLM 로드 및 PII 탐지 ---
    success, msg = model_manager.load_llm()
    if not success:
        return "LLM 로드 실패", "N/A", [], f"Error: {msg}"
    
    results = []
    target_categories = [
        ("FIRSTNAME", "Individual person's name"),
        ("ADDRESS", "Full address string including City and Street"),
        ("STATE", "Province, State, or Do name"),
        ("CITY", "City, District, or Si/Gun/Gu name"),
        ("STREET", "Road name or Street address"),
        ("ZIPCODE", "Postal or Zip code")
    ]

    for data in units_data:
        unit = data["text"]
        keywords = data["keywords"]
        
        # A. 정규식 탐지 (전화번호, 이메일)
        for m in PHONE_REGEX.finditer(unit): results.append({"word": m.group(), "type": "PHONE"})
        for m in EMAIL_REGEX.finditer(unit): results.append({"word": m.group(), "type": "EMAIL"})
        
        # B. LLM 탐지 (Stanza 힌트 전달)
        for label, desc in target_categories:
            found = extract_specific_pii(unit, label, desc, keywords=keywords)
            results.extend(found)

    # --- [Step 3] 결과 정제 및 마스킹 ---
    unique_results = []
    seen_words = set()
    for res in sorted(results, key=lambda x: len(str(x.get('word',''))), reverse=True):
        word = str(res.get('word', '')).strip()
        if word and word not in seen_words and len(word) > 1:
            unique_results.append(res)
            seen_words.add(word)

    masked_text = text
    final_entities = []
    address_mask_targets = ["ADDRESS", "STREET", "ZIPCODE", "BUILDINGNUMBER"]
    address_expose_targets = ["CITY", "STATE", "COUNTY"]

    for res in unique_results:
        word, p_type = res['word'], res['type']
        final_entities.append((word, p_type))
        
        if minimize_address and p_type in address_mask_targets:
            if word in masked_text:
                masked_text = masked_text.replace(word, mask_char * len(word))
        elif minimize_address and p_type in address_expose_targets:
            pass
        else:
            if word in masked_text:
                masked_text = masked_text.replace(word, mask_char * len(word))
            
    return masked_text, f"Main Lang: {main_lang}", final_entities, "✨ Hybrid Detection Complete"
