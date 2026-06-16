import re
import json
import torch
from model_manager import model_manager
from utils import get_best_language
from config import PII_CATEGORIES_MAP

# 최소한의 안전망 정규식
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{3,4}[- ]?\d{4}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def is_category_group_present(text, group_name, labels, keywords=None):
    """1단계: 특정 카테고리 그룹 존재 여부 체크 (Yes/No)"""
    if model_manager.model is None or not text.strip():
        return False

    hint_str = f"\nHint Keywords: {', '.join(keywords)}" if keywords else ""
    prompt = f"""<|start_of_role|>system<|end_of_role|>Does the text contain '{group_name}' info?
Possible: {", ".join(labels)}
Rules: Answer ONLY 'YES' or 'NO'.{hint_str}
<|start_of_role|>user<|end_of_role|>Text: {text}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
Answer:"""
    
    inputs = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.model.device)
    with torch.no_grad():
        outputs = model_manager.model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=False, pad_token_id=model_manager.tokenizer.eos_token_id)
    
    response = model_manager.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().upper()
    return "YES" in response

def validate_pii_format(word, p_type, original_text):
    """추출된 PII 형식 검증 (350M 모델 오탐 방어)"""
    word_clean = word.replace("-", "").replace(" ", "").replace("/", "").replace(".", "")
    
    if p_type in ["VIN", "VRM", "IMEI", "MACADDRESS", "IPADDRESS", "BITCOINADDRESS", "ETHEREUMADDRESS"]:
        if any('\uac00' <= c <= '\ud7a3' for c in word): return False
        if p_type == "VIN" and len(word_clean) < 10: return False
    
    if p_type == "DATE":
        # 지명 오탐 방지
        if word.endswith(("시", "구", "도", "군", "읍", "면", "동", "길", "로")): return False
        # 단순 숫자인 경우 날짜 키워드가 문맥에 있어야 함
        if word_clean.isdigit():
            context_keywords = ["년", "월", "일", "Date", "Born", "Expired"]
            if not any(k in original_text for k in context_keywords): return False
        if not any(c.isdigit() for c in word): return False

    if p_type == "ZIPCODE":
        if not word_clean.isdigit() or len(word_clean) < 5: return False

    return True

def extract_specific_pii(text, group_name, labels, keywords=None):
    """2단계: 실제 값 추출"""
    if model_manager.model is None or not text.strip():
        return []

    hint_str = f"\nKeywords: {', '.join(keywords)}" if keywords else ""
    prompt = f"""<|start_of_role|>system<|end_of_role|>Extract '{group_name}' PII.
Labels: {", ".join(labels)}
Rules: Return ONLY JSON array: [{{"word": "text", "type": "LABEL"}}]{hint_str}
<|start_of_role|>user<|end_of_role|>Text: {text}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
JSON:"""
    
    inputs = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.model.device)
    with torch.no_grad():
        outputs = model_manager.model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=False, pad_token_id=model_manager.tokenizer.eos_token_id)
    
    response = model_manager.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    blacklist = {"PII content", "Hello", "Thank you", "Notice", "Project", "My office", "located at", "문의", "contact"}
    
    try:
        match = re.search(r'\[.*\]', response.strip(), re.DOTALL)
        if match:
            items = json.loads(match.group())
            return [i for i in items if isinstance(i, dict) and str(i.get('word','')) in text and str(i.get('type','')).upper() in labels and str(i.get('word','')) not in blacklist and validate_pii_format(str(i.get('word','')), str(i.get('type','')).upper(), text)]
        return []
    except:
        return []

def process_pii(text, mask_char="*", minimize_address=False):
    if not text: return "입력 비었음", "N/A", [], "입력 대기"
    main_lang, _ = get_best_language(text)
    
    # [1] Stanza 키워드 추출
    nlp = model_manager.load_stanza(main_lang, processors='tokenize,pos')
    units_data = []
    if nlp:
        doc = nlp(text)
        for sent in doc.sentences:
            keywords = [w.text for w in sent.words if w.upos in ['PROPN', 'NOUN', 'NUM']]
            units_data.append({"text": sent.text, "keywords": list(set(keywords))})
        model_manager.unload_stanza()
    else:
        units_data = [{"text": line, "keywords": []} for line in text.split('\n') if line.strip()]

    # [2] LLM 로드
    success, msg = model_manager.load_llm()
    if not success: return "LLM 로드 실패", "N/A", [], f"Error: {msg}"
    
    final_entities = []
    working_text = text
    priority_order = ["Address", "Identity", "Contact", "Financial", "Dates & time", "Digital", "Vehicle", "Crypto"]

    # --- EXCLUSIVE SEQUENTIAL EXTRACTION ---
    for data in units_data:
        unit = data["text"]
        keywords = data["keywords"]
        
        # A. 정규식
        for m in PHONE_REGEX.finditer(unit): final_entities.append((m.group(), "PHONE"))
        for m in EMAIL_REGEX.finditer(unit): final_entities.append((m.group(), "EMAIL"))
        
        # B. 그룹별 순차 추출 (앞선 그룹이 찾은 영역은 제외하는 논리적 독점)
        for group_name in priority_order:
            labels = PII_CATEGORIES_MAP.get(group_name, [])
            # 이미 마스킹된 영역을 제외한 나머지 unit에서만 찾도록 (실제로는 추출 후 필터링)
            if is_category_group_present(unit, group_name, labels, keywords=keywords):
                found = extract_specific_pii(unit, group_name, labels, keywords=keywords)
                for item in found:
                    word, p_type = item['word'], item['type']
                    # 중복 방지 및 독점 로직
                    if not any(word in str(e[0]) for e in final_entities):
                        final_entities.append((word, p_type))

    # [3] 마스킹 적용 (가장 긴 단어부터)
    unique_entities = sorted(list(set(final_entities)), key=lambda x: len(str(x[0])), reverse=True)
    masked_text = text
    
    address_mask_targets = ["STREET", "ZIPCODE", "BUILDINGNUMBER", "ADDRESS"]
    address_expose_targets = ["CITY", "STATE", "COUNTY"]

    for word, p_type in unique_entities:
        if minimize_address and p_type in address_mask_targets:
            masked_text = masked_text.replace(word, mask_char * len(word))
        elif minimize_address and p_type in address_expose_targets:
            pass
        else:
            masked_text = masked_text.replace(word, mask_char * len(word))
            
    return masked_text, f"Lang: {main_lang}", unique_entities, "✨ Exclusive Detection Complete"
