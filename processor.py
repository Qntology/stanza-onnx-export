import re
import json
import threading
from model_manager import model_manager
from utils import get_best_language
from config import PII_CATEGORIES_MAP, ADDRESS_DICTIONARY, PHONE_REGEX_MAP, EMAIL_REGEX, PCC_REGEX, TRACKING_CANDIDATE_REGEX, NUMBER_BLOCK_REGEX

# 전역 처리 잠금 (순차 대기열 보장)
_processing_lock = threading.Lock()

def preprocess_text(text):
    """특수문자 주변에 공백을 삽입하여 단어 분리(Tokenization) 성능 향상.
    치환 후 원래대로 돌리기 위해 특수 마커를 사용할 수 있으나, 여기서는 단순히 공백을 추가함.
    최종 출력에서 공백이 생기는 문제를 방지하기 위해 나중에 역처리할 수도 있음."""
    if not text: return ""
    # 구두점 및 괄호 주변 공백 삽입 (단, 이미 공백이 있는 경우는 중복 방지)
    processed = re.sub(r'([.,!?()\[\]{}|/\\<>])', r' \1 ', text)
    processed = re.sub(r' +', ' ', processed)
    return processed.strip()

def postprocess_text(text):
    """전처리 시 추가된 불필요한 공백을 제거하여 원래 문장 부호 구조 복원"""
    if not text: return ""
    # 구두점 앞의 공백 제거
    text = re.sub(r' ([.,!?()\[\]{}|/\\<>])', r'\1', text)
    # 구두점 뒤의 공백은 유지하되, 연속된 공백은 하나로
    text = re.sub(r' +', ' ', text)
    return text.strip()

def verify_with_llm(word, p_type, context):
    if model_manager.model is None: return True
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
        context_area = text[max(0, start-20):min(len(text), end+20)].lower()
        
        if any(kw in context_area for kw in tracking_keywords):
            results.append({"word": num, "type": "TRACKING_NUMBER", "context": context_area})
    return results

def scan_address_candidates(text, lang_code):
    """앵커 키워드 기반으로 상세 주소 후보 수집 (정밀화)"""
    if lang_code not in ADDRESS_DICTIONARY: return []
    dict_info = ADDRESS_DICTIONARY[lang_code]
    anchors = dict_info["admin"] + dict_info["street"]
    results = []
    
    words = text.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w\s]', '', word)
        
        found_anchor = None
        for anchor in anchors:
            if clean_word.endswith(anchor) or (lang_code not in ['ko', 'ja', 'zh-hans'] and clean_word == anchor):
                found_anchor = anchor
                break
        
        if found_anchor:
            addr_parts = [word]
            # 뒤쪽 확인 (번지수, 건물번호 등) - 범위를 5단어로 확장
            for j in range(i+1, min(len(words), i+6)):
                next_word = words[j]
                # 숫자 포함, 앵커 키워드 포함, 혹은 매우 짧은 단어(번지 등)
                if any(c.isdigit() for c in next_word) or any(a in next_word for a in anchors) or len(next_word) <= 2:
                    addr_parts.append(next_word)
                else:
                    break
            # 앞쪽 확인 (시/도 등) - 범위를 5단어로 확장
            for j in range(i-1, max(-1, i-6), -1):
                prev_word = words[j]
                if any(a in prev_word for a in anchors):
                    addr_parts.insert(0, prev_word)
                else:
                    break
            full_word = " ".join(addr_parts)
            if len(full_word) > 2:
                results.append({"word": full_word, "type": "ADDRESS", "context": " ".join(words[max(0, i-3):min(len(words), i+4)])})
            
            zip_match = re.search(dict_info["zipcode_regex"], " ".join(words[max(0, i-4):min(len(words), i+5)]))
            if zip_match:
                results.append({"word": zip_match.group(), "type": "ZIPCODE", "context": " ".join(words[max(0, i-4):min(len(words), i+5)])})
    return results

def refine_entities_with_hint(text, found_results, main_lang):
    """Stanza의 의존성 구문 분석을 사용하여 인접 엔티티 결합"""
    nlp = model_manager.load_stanza(main_lang, processors='tokenize,pos,lemma,depparse')
    if not nlp: return found_results
    
    doc = nlp(text)
    refined = []
    tokens = []
    for sent in doc.sentences:
        tokens.extend(sent.words)
    model_manager.unload_stanza()
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        curr_text = token.text
        matching_ent = None
        for word, p_type in found_results:
            if curr_text in word:
                matching_ent = (word, p_type)
                break
        if matching_ent:
            j = i + 1
            combined_text = curr_text
            while j < len(tokens):
                next_token = tokens[j]
                if next_token.deprel in ['flat', 'compound', 'flat:name', 'nummod']:
                    combined_text += " " + next_token.text
                    j += 1
                else:
                    break
            if combined_text in text:
                refined.append((combined_text, matching_ent[1]))
            i = j
        else:
            i += 1
    return list(set(found_results + refined))

def finalize_contextual_labels(text, entities, main_lang):
    """문맥 기반 최종 라벨링 및 패턴 보정 (Linguistic-First)"""
    from config import LANG_REGISTRY, PHONE_REGEX_MAP, PCC_REGEX, EMAIL_REGEX
    final_entities = []
    
    # 중복 제거 및 길이순 정렬 (긴 것 우선)
    unique_words = sorted(list(set([e[0] for e in entities])), key=len, reverse=True)
    doc_lang_info = LANG_REGISTRY.get(main_lang, {"name": "Unknown", "prefix": "UN"})
    tag_prefix = doc_lang_info["prefix"]

    for word in unique_words:
        # 1. 고신뢰 패턴 (Regex Only)
        if EMAIL_REGEX.fullmatch(word):
            final_entities.append((word, f"{tag_prefix}_EMAIL"))
            continue
            
        phone_regex = PHONE_REGEX_MAP.get(main_lang, PHONE_REGEX_MAP["default"])
        if phone_regex.search(word):
            final_entities.append((word, f"{tag_prefix}_PHONE"))
            continue
            
        if PCC_REGEX.search(word):
            final_entities.append((word, f"{tag_prefix}_PCC"))
            continue

        # 2. 복합 엔티티 (Address, Person, Org) - LLM 검증 및 분류
        idx = text.find(word)
        if idx == -1: continue
        context = text[max(0, idx-50):min(len(text), idx+len(word)+50)]
        
        # LLM에게 카테고리와 '마스킹 시작 지점'을 묻는 정교한 프롬프트
        prompt = f"""<|start_of_role|>system<|end_of_role|>Task: PII Classification & redacting logic
Context: {context}
Identify the category of '{word}'. 
For Address, split into 'ADMIN' (City/Province/District) and 'DETAIL' (Street/Number/Unit).
Return format: CATEGORY: [LABEL] | PARTS: [Optional split for Address]
Labels: PERSON, ORGANIZATION, LOCATION, ADDRESS, NONE
<|start_of_role|>user<|end_of_role|>Word: {word}
<|start_of_role|>assistant<|end_of_role|>
Result:"""
        try:
            model_manager.model.reset()
            res = model_manager.model.create_completion(prompt=prompt, max_tokens=50, temperature=0)['choices'][0]['text'].strip().upper()
            
            if "NONE" in res: continue

            res_label = "NONE"
            for lbl in ["ORGANIZATION", "LOCATION", "ADDRESS", "PERSON"]:
                if lbl in res:
                    res_label = lbl
                    break
            
            if res_label == "ADDRESS":
                # 주소인 경우 상세 파트 추출 시도
                if "PARTS:" in res:
                    parts_info = res.split("PARTS:")[1].strip()
                    final_entities.append((word, f"{tag_prefix}_ADDRESS_FULL|{parts_info}"))
                else:
                    final_entities.append((word, f"{tag_prefix}_ADDRESS"))
            elif res_label != "NONE":
                final_entities.append((word, f"{tag_prefix}_{res_label}"))
        except:
            # 에러 시 기본 타입 유지
            original_type = next((e[1] for e in entities if e[0] == word), "PROPN")
            final_entities.append((word, f"{tag_prefix}_{original_type}"))
            
    return final_entities

def process_pii(text, mask_char="*", minimize_address=False):
    with _processing_lock:
        if not text: return "Empty", "N/A", [], "Idle"
        
        # 1. 전처리 (Tokenization 향상)
        clean_text = preprocess_text(text)
        main_lang, _ = get_best_language(clean_text)
        
        # 2. 언어학적 후보군 추출 (Stanza)
        nlp = model_manager.load_stanza(main_lang, processors='tokenize,pos,lemma,depparse,ner' if main_lang in ['en', 'zh-hans'] else 'tokenize,pos,lemma,depparse')
        if not nlp: return "Error", "N/A", [], "Stanza Load Failed"
        
        doc = nlp(clean_text)
        candidates = []
        
        # 앵커 기반 주소 스캔 및 구문 분석 활용
        for sent in doc.sentences:
            # (1) 주소 앵커 및 숫자 패턴
            addr_cands = scan_address_candidates(sent.text, main_lang)
            for c in addr_cands:
                candidates.append((c["word"], "ADDRESS"))
            
            # (2) Stanza 고유명사 및 NER
            if hasattr(sent, 'entities'):
                for ent in sent.entities:
                    candidates.append((ent.text, "PROPN"))
            
            for word in sent.words:
                if word.upos in ['PROPN', 'X']:
                    candidates.append((word.text, "PROPN"))
        
        # (3) 강력한 패턴 (전화, 이메일 등) 직접 추가
        for m in EMAIL_REGEX.finditer(clean_text):
            candidates.append((m.group(), "EMAIL"))
        phone_regex = PHONE_REGEX_MAP.get(main_lang, PHONE_REGEX_MAP["default"])
        for m in phone_regex.finditer(clean_text):
            candidates.append((m.group(), "PHONE"))
        for m in PCC_REGEX.finditer(clean_text):
            candidates.append((m.group(), "PCC"))

        model_manager.unload_stanza()
        
        # 3. LLM 검증 및 최종 라벨링
        success, msg = model_manager.load_llm()
        if not success: return "Error", "N/A", [], msg
        
        # 중복 제거 및 구문 기반 정교화
        unique_cands = list(set(candidates))
        refined_cands = refine_entities_with_hint(clean_text, unique_cands, main_lang)
        final_entities = finalize_contextual_labels(clean_text, refined_cands, main_lang)
        
        # 4. 마스킹 (Policy-based)
        masked_text = clean_text
        # 긴 단어부터 치환하여 부분 포함 관계 오류 방지
        unique_entities = sorted(list(set(final_entities)), key=lambda x: len(str(x[0])), reverse=True)
        
        for word, full_type in unique_entities:
            if len(word) < 2 and "PHONE" not in full_type and "EMAIL" not in full_type: continue
            
            replacement = mask_char * len(word)
            
            # 주소 최소화 정책 적용
            if minimize_address and "ADDRESS" in full_type:
                if "|" in full_type:
                    # LLM이 구분해준 파트 활용
                    try:
                        detail_part = full_type.split("|")[1].strip()
                        if detail_part in word and len(detail_part) > 1:
                            # 상세 주소 부분만 마스킹
                            replacement = word.replace(detail_part, mask_char * len(detail_part))
                    except: pass
                else:
                    # 기본 규칙: 공백 기준 뒤쪽 파트 마스킹
                    parts = word.split()
                    if len(parts) > 2:
                        replacement = " ".join(parts[:2]) + " " + " ".join([mask_char * len(p) for p in parts[2:]])
            
            masked_text = masked_text.replace(word, replacement)
            
        return postprocess_text(masked_text), f"Lang: {main_lang}", unique_entities, "PII Tiered Processing Complete"
