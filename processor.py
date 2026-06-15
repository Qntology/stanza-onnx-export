import re
import json
import torch
from config import PII_CATEGORIES
from model_manager import model_manager
from utils import get_best_language

# 정규표현식 패턴 - 좀 더 구체적으로 조정 (주문번호 등 오탐 방지)
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{3,4}[- ]?\d{4}')
# 이메일은 비교적 정확하므로 유지
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def extract_pii_with_llm(text, lang_code):
    """LLM을 사용하여 PII 추출 및 분류 (350M 모델에 최적화된 프롬프트)"""
    if model_manager.model is None:
        return []

    # 모델의 부담을 줄이기 위해 가장 중요한 카테고리 위주로 예시와 함께 강조
    prompt = f"""<|start_of_role|>system<|end_of_role|>You are a PII detection bot. Extract PII from the text.
Available Labels: 
- Identity: FIRSTNAME, LASTNAME, MIDDLENAME, AGE, JOBTITLE, ORGANIZATION
- Contact: EMAIL, PHONE
- Address: STREET, CITY, COUNTY, STATE, ZIPCODE
- Financial: BANKACCOUNT, CREDITCARD, AMOUNT

Rules:
1. Return ONLY a JSON array: [{{"word": "text", "type": "LABEL"}}]
2. For Korean addresses: "서울특별시" is CITY, "서초구" is COUNTY, "반포대로 23길 45" is STREET.
3. DO NOT extract "Hello", "Thank you", "Notice", "Project" or common nouns.
4. If no PII, return [].
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
            found_items = json.loads(match.group())
            if isinstance(found_items, list):
                # 유효한 카테고리만 허용
                return [{"word": str(item['word']), "type": str(item['type']).upper()} 
                        for item in found_items 
                        if 'word' in item and 'type' in item]
        return []
    except:
        return []

def process_pii(text, mask_char="*", minimize_address=False):
    if not text:
        return "입력 텍스트가 비어 있습니다.", "N/A", [], "입력 대기 중"

    # 1. 언어 감지
    main_lang, sub_langs = get_best_language(text)
    
    # 2. 자동 다운로드 체크
    download_msg = ""
    if not model_manager.is_stanza_installed(main_lang):
        download_msg = f"🔄 {main_lang} 모델 자동 다운로드 중..."
        success, msg = model_manager.download_stanza_model(main_lang)
        download_msg = f"✅ {main_lang} 다운로드 완료!" if success else f"❌ {main_lang} 실패: {msg}"
    else:
        download_msg = f"✨ {main_lang} 모델 준비 완료."

    # --- [VRAM STEP 1] Stanza 로드 및 텍스트 분할 ---
    nlp = model_manager.load_stanza(main_lang)
    units = []
    pos_info = f"Main Language: {main_lang}\n"
    
    if nlp:
        doc = nlp(text)
        units = [sent.text for sent in doc.sentences]
        model_manager.unload_stanza()
    else:
        units = [line.strip() for line in text.split('\n') if line.strip()]

    # --- [VRAM STEP 2] LLM 로드 및 PII 탐지 ---
    success, msg = model_manager.load_llm()
    if not success:
        return "LLM 로드 실패", "N/A", [], f"Error: {msg}"
    
    results = []
    for i, unit in enumerate(units):
        # Regex (Safety Net)
        for m in PHONE_REGEX.finditer(unit): results.append({"word": m.group(), "type": "PHONE"})
        for m in EMAIL_REGEX.finditer(unit): results.append({"word": m.group(), "type": "EMAIL"})
        
        # LLM 탐지 (개선된 프롬프트)
        llm_found = extract_pii_with_llm(unit, main_lang)
        results.extend(llm_found)

    # 4. 결과 정제
    unique_results = []
    seen_words = set()
    for res in sorted(results, key=lambda x: len(x.get('word','')), reverse=True):
        word = res.get('word', '')
        if word and word not in seen_words:
            unique_results.append(res)
            seen_words.add(word)

    masked_text = text
    final_entities = []

    # 주소 최소화 시 마스킹 대상
    address_mask_targets = ["STREET", "BUILDINGNUMBER", "SECONDARYADDRESS", "ZIPCODE", "GPSCOORDINATES"]
    address_expose_targets = ["CITY", "STATE", "COUNTY"]

    for res in unique_results:
        word = res['word']
        p_type = res['type']
        
        # 주소 최소화 로직 적용
        if minimize_address and p_type in address_mask_targets:
            final_entities.append((word, p_type))
            if word in masked_text:
                masked_text = masked_text.replace(word, mask_char * len(word))
        elif minimize_address and p_type in address_expose_targets:
            final_entities.append((word, p_type))
            # 마스킹 하지 않음
        else:
            final_entities.append((word, p_type))
            if word in masked_text:
                if mask_char == "[TAG]":
                    masked_text = masked_text.replace(word, f"[{p_type}]")
                else:
                    masked_text = masked_text.replace(word, mask_char * len(word))
            
    return masked_text, pos_info, final_entities, download_msg
