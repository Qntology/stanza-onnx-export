import re
import json
from model_manager import model_manager
from utils import get_best_language
from config import PII_CATEGORIES_MAP, ADDRESS_DICTIONARY

# 1. 정규표현식 정의
# 핵심 PII (선점 대상)
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{3,4}[- ]?\d{4}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# [신규] 개인통관고유부호 (P + 12자리 숫자)
PCC_REGEX = re.compile(r'\bP\d{12}\b', re.IGNORECASE)

# [신규] 운송장번호 후보 (9~14자리 연속 숫자)
TRACKING_CANDIDATE_REGEX = re.compile(r'\b\d{9,14}\b')

# 숫자 블록 패턴 (번지수 등)
NUMBER_BLOCK_REGEX = re.compile(r'^[\d\-\./]+$')

def verify_with_llm(word, p_type, context):
    if model_manager.model is None: return True
    prompt = f"""<|start_of_role|>system<|end_of_role|>Is '{word}' a valid '{p_type}'?
Context: {context}
Rules: Answer ONLY 'YES' or 'NO'.
<|start_of_role|>user<|end_of_role|>Word: {word}
<|start_of_role|>assistant<|end_of_role|>
Answer:"""
    try:
        model_manager.model.reset()
        res = model_manager.model.create_completion(prompt=prompt, max_tokens=5, temperature=0)['choices'][0]['text'].strip().upper()
        return "YES" in res
    except: return True

def scan_tracking_numbers(text):
    """문맥 기반 운송장번호 탐지"""
    results = []
    tracking_keywords = ["운송장", "송장", "tracking", "waybill", "배송번호", "주문번호"]
    
    for match in TRACKING_CANDIDATE_REGEX.finditer(text):
        num = match.group()
        start, end = match.start(), match.end()
        # 주변 문맥 (좌우 20자) 확인
        context_area = text[max(0, start-20):min(len(text), end+20)].lower()
        
        if any(kw in context_area for kw in tracking_keywords):
            results.append({"word": num, "type": "TRACKING_NUMBER", "context": context_area})
    return results

def scan_address_candidates(text, lang_code):
    if lang_code not in ADDRESS_DICTIONARY: return []
    dict_info = ADDRESS_DICTIONARY[lang_code]
    anchors = dict_info["admin"] + dict_info["street"]
    results = []
    
    ko_blacklist = {"있으므로", "반드시", "함으로", "하므로", "함으로서", "도시", "당시", "동시"}
    words = text.split()
    
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w\s]', '', word)
        if lang_code == "ko" and clean_word in ko_blacklist: continue
        
        found_anchor = None
        for anchor in anchors:
            if clean_word.endswith(anchor) or (lang_code != 'ko' and clean_word == anchor):
                found_anchor = anchor
                break
        
        if found_anchor:
            final_address_block = [word]
            if i > 0 and NUMBER_BLOCK_REGEX.match(words[i-1]):
                final_address_block.insert(0, words[i-1])
            if i < len(words) - 1 and NUMBER_BLOCK_REGEX.match(re.sub(r'[^\w\d\-\./]', '', words[i+1])):
                final_address_block.append(words[i+1])
            
            full_word = " ".join(final_address_block)
            context = " ".join(words[max(0, i-3):min(len(words), i+4)])
            results.append({"word": full_word, "type": "ADDRESS", "context": context})
            
            zip_match = re.search(dict_info["zipcode_regex"], context)
            if zip_match:
                results.append({"word": zip_match.group(), "type": "ZIPCODE", "context": context})
    return results

def classify_pii_with_pos(word, pos_tag, context):
    tag_map = {
        'PROPN': ["FIRSTNAME", "LASTNAME", "ORGANIZATION"],
        'NUM': ["AGE", "BANKACCOUNT", "CREDITCARD", "SSN", "DATE"],
        'NOUN': ["JOBTITLE", "OCCUPATION"]
    }
    target_labels = tag_map.get(pos_tag, ["FIRSTNAME", "ORGANIZATION"])
    prompt = f"""<|start_of_role|>system<|end_of_role|>Classify: {word}
Context: {context}
Labels: {", ".join(target_labels)}
Rules: Return ONLY LABEL or 'NONE'.
<|start_of_role|>user<|end_of_role|>Word: {word}
<|start_of_role|>assistant<|end_of_role|>
Label:"""
    try:
        model_manager.model.reset()
        res = model_manager.model.create_completion(prompt=prompt, max_tokens=10, temperature=0)['choices'][0]['text'].strip().upper()
        for lbl in target_labels:
            if lbl in res: return lbl
        return None
    except: return None

def process_pii(text, mask_char="*", minimize_address=False):
    if not text: return "Empty", "N/A", [], "Idle"
    main_lang, _ = get_best_language(text)
    
    # [1] 다양한 후보군 추출
    address_candidates = scan_address_candidates(text, main_lang)
    tracking_candidates = scan_tracking_numbers(text)
    
    nlp = model_manager.load_stanza(main_lang, processors='tokenize,pos')
    pos_candidates = []
    if nlp:
        doc = nlp(text)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos in ['PROPN', 'NUM', 'X']:
                    pos_candidates.append({"word": word.text, "upos": word.upos, "context": sent.text})
        model_manager.unload_stanza()
    
    success, msg = model_manager.load_llm()
    if not success: return "Error", "N/A", [], msg
    
    final_results = []
    # 1. 고정 정규식 선점 (Phone, Email, PCC)
    for m in PHONE_REGEX.finditer(text): final_results.append((m.group(), "PHONE"))
    for m in EMAIL_REGEX.finditer(text): final_results.append((m.group(), "EMAIL"))
    for m in PCC_REGEX.finditer(text): final_results.append((m.group(), "PCC"))
    
    # 2. 주소 및 운송장번호 선점
    for cand in address_candidates + tracking_candidates:
        if verify_with_llm(cand["word"], cand["type"], cand["context"]):
            if not any(cand["word"] in str(r[0]) for r in final_results):
                final_results.append((cand["word"], cand["type"]))
            
    # 3. 기타 POS 후보
    for cand in pos_candidates:
        word = cand["word"]
        if any(word in str(r[0]) for r in final_results): continue
        label = classify_pii_with_pos(word, cand["upos"], cand["context"])
        if label: final_results.append((word, label))

    # [4] 마스킹
    unique_entities = sorted(list(set(final_results)), key=lambda x: len(str(x[0])), reverse=True)
    masked_text = text
    for word, p_type in unique_entities:
        if minimize_address:
            if p_type in ["STREET", "ZIPCODE", "ADDRESS", "PCC", "TRACKING_NUMBER"]:
                masked_text = masked_text.replace(word, mask_char * len(word))
            elif any(word.endswith(s) for s in ["구", "군", "읍", "면", "동"]):
                masked_text = masked_text.replace(word, mask_char * len(word))
        else:
            masked_text = masked_text.replace(word, mask_char * len(word))
            
    return masked_text, f"Lang: {main_lang}", unique_entities, "✨ Logistic PII Detection Complete"
