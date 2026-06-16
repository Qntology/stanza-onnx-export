import re
import json
import threading
from model_manager import model_manager
from utils import get_best_language
from config import PII_CATEGORIES_MAP, ADDRESS_DICTIONARY

# 전역 처리 잠금 (순차 대기열 보장)
_processing_lock = threading.Lock()

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

def preprocess_text(text):
    """특수문자 주변에 공백을 삽입하여 단어 분리(Tokenization) 성능 향상"""
    if not text: return ""
    # 주요 구분 기호 주변 공백 삽입: [], (), {}, |, /, \, <, >, |
    symbols = r'([\[\]\(\)\{\}\|\/\\<>])'
    processed = re.sub(symbols, r' \1 ', text)
    # 중복 공백 제거
    processed = re.sub(r' +', ' ', processed)
    return processed.strip()

def verify_with_llm(word, p_type, context):
    if model_manager.model is None: return True
    # p_type이 PROPN 등 기술적 용어일 경우 일반적인 'Entity'로 설명
    display_type = "PII Entity" if p_type == "PROPN" else p_type
    
    prompt = f"""<|start_of_role|>system<|end_of_role|>Task: PII Verification
Question: Is '{word}' a valid '{display_type}' in the given context?
Context: ...{context}...
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

def refine_entities_with_hint(text, found_results, main_lang):
    """탐지된 이름/주소를 HINT로 사용하여 주변 누락분 및 상세 정보를 2차 확장 탐지"""
    from config import LANG_REGISTRY
    lang_info = LANG_REGISTRY.get(main_lang, {"name": "Unknown", "prefix": "UN"})
    tag_prefix = lang_info["prefix"]
    
    # 정제 대상: PERSON, ADDRESS
    targets = [r for r in found_results if any(t in r[1] for t in ["PERSON", "ADDRESS", "LOCATION"])]
    new_finds = []
    
    for word, p_type in targets:
        idx = text.find(word)
        if idx == -1: continue
        # 주변 컨텍스트 (앞뒤 30자)
        context_window = text[max(0, idx-30):min(len(text), idx+len(word)+30)]
        
        hint_type = "PERSON name" if "PERSON" in p_type else "ADDRESS/LOCATION"
        prompt = f"""<|start_of_role|>system<|end_of_role|>Task: Entity Expansion
HINT: '{word}' is part of a {hint_type} in {lang_info['name']}.
Context: ...{context_window}...
Question: Find the FULL range of this {hint_type}. (e.g., Full Name or Full Address)
Rules: Return ONLY the full string found in context.
<|start_of_role|>user<|end_of_role|>Context: {context_window}
<|start_of_role|>assistant<|end_of_role|>
Full Entity:"""
        
        try:
            model_manager.model.reset()
            res = model_manager.model.create_completion(prompt=prompt, max_tokens=40, temperature=0)['choices'][0]['text'].strip()
            if res and len(res) > 1 and res in context_window:
                new_finds.append((res, p_type)) # 원래 타입을 유지하되 범위 확장
        except: continue
        
    return list(set(found_results + new_finds))

def finalize_contextual_labels(text, entities, main_lang):
    """[4단계] 수집/확장된 모든 엔티티에 대해 문맥 기반 최종 라벨링 (스포츠팀 vs 지명 등)"""
    from config import LANG_REGISTRY
    final_entities = []
    
    unique_words = sorted(list(set([e[0] for e in entities])), key=len, reverse=True)
    
    for word in unique_words:
        # 단어의 지배적 언어 재확인
        from utils import get_best_language
        word_lang, _ = get_best_language(word)
        lang_info = LANG_REGISTRY.get(word_lang, LANG_REGISTRY.get(main_lang, {"name": "Unknown", "prefix": "UN"}))
        tag_prefix = lang_info["prefix"]
        
        # 문맥 추출 (충분히 넓게)
        idx = text.find(word)
        context = text[max(0, idx-40):min(len(text), idx+len(word)+40)]
        
        prompt = f"""<|start_of_role|>system<|end_of_role|>Task: PII Classification
Context: ...{context}...
Word: '{word}'
Question: In this context, is '{word}' a:
1. PERSON (Name)
2. ORGANIZATION (Team, Company, Club)
3. LOCATION (City, Country)
4. ADDRESS (Street, Building)
5. NONE (Not PII)

Rules: Return ONLY the category name. If unsure but it looks like a Proper Noun, choose PERSON or ORGANIZATION.
<|start_of_role|>user<|end_of_role|>Word: {word}
<|start_of_role|>assistant<|end_of_role|>
Category:"""
        try:
            model_manager.model.reset()
            res = model_manager.model.create_completion(prompt=prompt, max_tokens=15, temperature=0)['choices'][0]['text'].strip().upper()
            
            # 매칭 로직 개선
            matched = "NONE"
            for lbl in ["PERSON", "ORGANIZATION", "LOCATION", "ADDRESS", "PHONE", "EMAIL", "PCC", "TRACKING_NUMBER", "AGE"]:
                if lbl in res:
                    matched = lbl
                    break
            
            if matched != "NONE":
                final_entities.append((word, f"{tag_prefix}_{matched}"))
        except:
            # 실패 시 기본적으로 PERSON으로 간주하여 마스킹 누락 방지 (보수적 접근)
            final_entities.append((word, f"{tag_prefix}_PERSON"))
            
    return final_entities

def process_pii(text, mask_char="*", minimize_address=False):
    with _processing_lock:
        if not text: return "Empty", "N/A", [], "Idle"
        
        # [Step 1] 전처리
        text = preprocess_text(text)
        main_lang, _ = get_best_language(text)
        
        # [Step 2] 후보군 수집 (Collection)
        candidates = []
        # 2.1 정규식 (Phone, Email, PCC)
        for m in PHONE_REGEX.finditer(text): candidates.append((m.group(), "PHONE"))
        for m in EMAIL_REGEX.finditer(text): candidates.append((m.group(), "EMAIL"))
        for m in PCC_REGEX.finditer(text): candidates.append((m.group(), "PCC"))
        
        # 2.2 주소/운송장
        for c in scan_address_candidates(text, main_lang) + scan_tracking_numbers(text):
            candidates.append((c["word"], c["type"]))
            
        # 2.3 POS(형태소) 분석 - 고유명사 및 숫자
        nlp = model_manager.load_stanza(main_lang, processors='tokenize,pos')
        if nlp:
            doc = nlp(text)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos in ['PROPN', 'X']: # 고유명사 위주
                        candidates.append((word.text, "PROPN"))
                    elif word.upos == 'NUM' and len(word.text) > 1: # 숫자 (나이 등)
                        candidates.append((word.text, "NUM"))
            model_manager.unload_stanza()
            
        # [Step 3] 확장 및 정제 (Refinement)
        success, msg = model_manager.load_llm()
        if not success: return "Error", "N/A", [], msg
        
        # 1차 중복 제거 (타입 무관하게 텍스트 위주)
        initial_cands = list(set(candidates))
        
        # HINT 기반 2차 확장 (이름/주소 범위 넓히기)
        refined_cands = refine_entities_with_hint(text, initial_cands, main_lang)
        
        # [Step 4] 문맥 기반 최종 라벨 확정 (Contextual Labeling)
        # 사용자 요청: 라벨링을 먼저 한 후 마스킹 진행
        final_entities = finalize_contextual_labels(text, refined_cands, main_lang)

        # [Step 5] 최종 마스킹 (Masking)
        unique_entities = sorted(list(set(final_entities)), key=lambda x: len(str(x[0])), reverse=True)
        masked_text = text
        for word, p_type in unique_entities:
            # 마스킹 예외: 너무 짧은 단어는 제외 (단, 정규식 매칭 제외)
            if len(word) < 2 and not any(t in p_type for t in ["PHONE", "EMAIL", "PCC"]): continue
            
            if minimize_address:
                if any(t in p_type for t in ["STREET", "ZIPCODE", "ADDRESS", "PCC", "TRACKING_NUMBER"]):
                    masked_text = masked_text.replace(word, mask_char * len(word))
                elif any(word.endswith(s) for s in ["구", "군", "읍", "면", "동"]):
                    masked_text = masked_text.replace(word, mask_char * len(word))
            else:
                masked_text = masked_text.replace(word, mask_char * len(word))
                
        return masked_text, f"Lang: {main_lang}", unique_entities, "✨ Logistic PII Detection Complete"
