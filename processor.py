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
    """[개편] 하드코딩 블랙리스트를 제거하고 앵커 기반으로 후보만 수집 (검증은 LLM에 위임)"""
    if lang_code not in ADDRESS_DICTIONARY: return []
    dict_info = ADDRESS_DICTIONARY[lang_code]
    anchors = dict_info["admin"] + dict_info["street"]
    results = []
    
    words = text.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w\s]', '', word)
        
        found_anchor = None
        for anchor in anchors:
            if clean_word.endswith(anchor) or (lang_code != 'ko' and clean_word == anchor):
                found_anchor = anchor
                break
        
        if found_anchor:
            # 앵커 주변 블록 수집
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
    """[신규] Stanza의 의존성 구문 분석(Dependency)을 사용하여 인접 엔티티 결합 및 경계 확정"""
    # 1. Stanza 재로드 (의존성 분석용)
    nlp = model_manager.load_stanza(main_lang, processors='tokenize,pos,lemma,depparse')
    if not nlp: return found_results
    
    doc = nlp(text)
    refined = []
    tokens = []
    for sent in doc.sentences:
        tokens.extend(sent.words)

    model_manager.unload_stanza()
    
    # 2. 결합 로직
    i = 0
    while i < len(tokens):
        token = tokens[i]
        curr_text = token.text
        
        # 현재 토큰이 엔티티의 일부인 경우
        matching_ent = None
        for word, p_type in found_results:
            if curr_text in word:
                matching_ent = (word, p_type)
                break
        
        if matching_ent:
            # 인접한 flat/compound 관계 토큰 찾기
            j = i + 1
            combined_text = curr_text
            while j < len(tokens):
                next_token = tokens[j]
                if next_token.deprel in ['flat', 'compound', 'flat:name', 'nummod']:
                    combined_text += " " + next_token.text
                    j += 1
                else:
                    break
            
            # 결합된 텍스트가 원본에 존재하는지 확인
            if combined_text in text:
                refined.append((combined_text, matching_ent[1]))
            i = j
        else:
            i += 1
            
    return list(set(found_results + refined))

def finalize_contextual_labels(text, entities, main_lang):
    """[4단계] 수집/확장된 모든 엔티티에 대해 문맥 기반 최종 라벨링 (스포츠팀 vs 지명 등)"""
    from config import LANG_REGISTRY
    final_entities = []
    
    unique_words = sorted(list(set([e[0] for e in entities])), key=len, reverse=True)
    
    # 카테고리 맵핑 보조 (이미 분류된 경우 LLM 생략 가능하게)
    pre_classified = {word: p_type for word, p_type in entities if p_type in ["PHONE", "EMAIL", "PCC", "TRACKING_NUMBER", "ZIPCODE"]}

    for word in unique_words:
        # 이미 정규식 등으로 확정된 타입은 그대로 사용
        if word in pre_classified:
            from utils import get_best_language
            word_lang, _ = get_best_language(word)
            lang_info = LANG_REGISTRY.get(word_lang, LANG_REGISTRY.get(main_lang, {"name": "Unknown", "prefix": "UN"}))
            final_entities.append((word, f"{lang_info['prefix']}_{pre_classified[word]}"))
            continue

        # 단어의 지배적 언어 재확인
        from utils import get_best_language
        word_lang, _ = get_best_language(word)
        lang_info = LANG_REGISTRY.get(word_lang, LANG_REGISTRY.get(main_lang, {"name": "Unknown", "prefix": "UN"}))
        tag_prefix = lang_info["prefix"]
        
        # 문맥 추출
        idx = text.find(word)
        if idx == -1: continue
        context = text[max(0, idx-50):min(len(text), idx+len(word)+50)]
        
        prompt = f"""<|start_of_role|>system<|end_of_role|>Task: PII Classification
Context: ...{context}...
Word: '{word}'
Question: In this context, is '{word}' a:
1. PERSON (Name of a person)
2. ORGANIZATION (Company, Team, Club, Agency)
3. LOCATION (City, Country, Region)
4. ADDRESS (Street name, Building name/number, Apartment)
5. PHONE (Phone number)
6. EMAIL (Email address)
7. NONE (Not a PII entity)

Rules: Return ONLY the category name. If it's a specific number like a tracking number or ID, choose NONE if not in list.
<|start_of_role|>user<|end_of_role|>Word: {word}
<|start_of_role|>assistant<|end_of_role|>
Category:"""
        try:
            model_manager.model.reset()
            res = model_manager.model.create_completion(prompt=prompt, max_tokens=15, temperature=0)['choices'][0]['text'].strip().upper()
            
            # 매칭 로직
            matched = "NONE"
            for lbl in ["PERSON", "ORGANIZATION", "LOCATION", "ADDRESS", "PHONE", "EMAIL", "PCC", "TRACKING_NUMBER", "AGE"]:
                if lbl in res:
                    matched = lbl
                    break
            
            if matched != "NONE":
                final_entities.append((word, f"{tag_prefix}_{matched}"))
        except:
            # 실패 시 보수적으로 처리
            final_entities.append((word, f"{tag_prefix}_PERSON"))
            
    return final_entities

def process_pii(text, mask_char="*", minimize_address=False):
    with _processing_lock:
        if not text: return "Empty", "N/A", [], "Idle"
        
        # [Step 1] 전처리
        text = preprocess_text(text)
        main_lang, _ = get_best_language(text)
        
        # [Step 2] Stanza를 이용한 문장 분할 및 POS/Dependency 분석 (고도화)
        nlp = model_manager.load_stanza(main_lang)
        if not nlp: return "Error", "N/A", [], "Stanza Load Failed"
        
        doc = nlp(text)
        all_candidates = []
        
        # 문장 단위로 독립 처리
        for sent in doc.sentences:
            sent_text = sent.text
            sent_candidates = []
            
            # 2.1 정규식 매칭 (문장 내)
            for m in PHONE_REGEX.finditer(sent_text): sent_candidates.append((m.group(), "PHONE"))
            for m in EMAIL_REGEX.finditer(sent_text): sent_candidates.append((m.group(), "EMAIL"))
            for m in PCC_REGEX.finditer(sent_text): sent_candidates.append((m.group(), "PCC"))
            
            # 2.2 주소/운송장 (문장 내)
            for c in scan_address_candidates(sent_text, main_lang) + scan_tracking_numbers(sent_text):
                sent_candidates.append((c["word"], c["type"]))
                
            # 2.3 POS 기반 고유명사
            for word in sent.words:
                if word.upos in ['PROPN', 'X']:
                    sent_candidates.append((word.text, "PROPN"))
                elif word.upos == 'NUM' and len(word.text) > 1:
                    sent_candidates.append((word.text, "NUM"))
            
            all_candidates.extend(sent_candidates)

        model_manager.unload_stanza()
        
        # [Step 3] LLM 추론 (분산 처리된 후보군 검증 및 확장)
        success, msg = model_manager.load_llm()
        if not success: return "Error", "N/A", [], msg
        
        # 1차 검증
        unique_cands = list(set(all_candidates))
        verified_cands = []
        for word, p_type in unique_cands:
            if p_type in ["PHONE", "EMAIL", "PCC"]:
                verified_cands.append((word, p_type))
            else:
                idx = text.find(word)
                win_text = text[max(0, idx-40):min(len(text), idx+len(word)+40)]
                if verify_with_llm(word, p_type, win_text):
                    verified_cands.append((word, p_type))
        
        # HINT 기반 2차 확장 (LLM)
        refined_llm = refine_entities_with_hint(text, verified_cands, main_lang)
        
        # [Step 4] 문맥 기반 최종 라벨링 및 마스킹
        final_entities = finalize_contextual_labels(text, refined_llm, main_lang)
        
        unique_entities = sorted(list(set(final_entities)), key=lambda x: len(str(x[0])), reverse=True)
        masked_text = text
        for word, p_type in unique_entities:
            if len(word) < 2 and not any(t in p_type for t in ["PHONE", "EMAIL", "PCC"]): continue
            
            if minimize_address:
                if any(t in p_type for t in ["STREET", "ZIPCODE", "ADDRESS", "PCC", "TRACKING_NUMBER"]):
                    masked_text = masked_text.replace(word, mask_char * len(word))
                elif any(word.endswith(s) for s in ["구", "군", "읍", "면", "동"]):
                    masked_text = masked_text.replace(word, mask_char * len(word))
            else:
                masked_text = masked_text.replace(word, mask_char * len(word))
                
        return masked_text, f"Lang: {main_lang}", unique_entities, "Linguistic PII Detection Complete"
