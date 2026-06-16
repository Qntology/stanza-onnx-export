import re
import json
import threading
from model_manager import model_manager
from utils import get_best_language
from config import PII_CATEGORIES_MAP, ADDRESS_DICTIONARY, PHONE_REGEX_MAP, EMAIL_REGEX, PCC_REGEX, TRACKING_CANDIDATE_REGEX, NUMBER_BLOCK_REGEX

# 전역 처리 잠금 (순차 대기열 보장)
_processing_lock = threading.Lock()

def preprocess_text(text):
    """특수문자 주변에 공백을 삽입하여 단어 분리(Tokenization) 성능 향상"""
    if not text: return ""
    symbols = r'([\[\]\(\)\{\}\|\/\\<>])'
    processed = re.sub(symbols, r' \1 ', text)
    processed = re.sub(r' +', ' ', processed)
    return processed.strip()

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
            # 뒤쪽 확인 (번지수, 건물번호 등)
            for j in range(i+1, min(len(words), i+3)):
                next_word = words[j]
                if any(c.isdigit() for c in next_word) or any(a in next_word for a in anchors):
                    addr_parts.append(next_word)
                else:
                    break
            # 앞쪽 확인 (시/도 등)
            for j in range(i-1, max(-1, i-2), -1):
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
    """문맥 기반 최종 라벨링 및 패턴 보정 (Pattern-First)"""
    from config import LANG_REGISTRY, PHONE_REGEX_MAP, PCC_REGEX, EMAIL_REGEX, TRACKING_CANDIDATE_REGEX
    final_entities = []
    unique_words = sorted(list(set([e[0] for e in entities])), key=len, reverse=True)
    doc_lang_info = LANG_REGISTRY.get(main_lang, {"name": "Unknown", "prefix": "UN"})
    tag_prefix = doc_lang_info["prefix"]

    for word in unique_words:
        # 1. 강력한 패턴 매칭 (LLM 불필요)
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

        # 2. 기타 고유명사/주소는 LLM 문맥 확인
        idx = text.find(word)
        if idx == -1: continue
        context = text[max(0, idx-40):min(len(text), idx+len(word)+40)]
        
        prompt = f"""<|start_of_role|>system<|end_of_role|>Task: PII Classification
Context: {context}
Identify if '{word}' is a Person, Organization, Location, or Address.
Return 'NONE' if it is not PII.
<|start_of_role|>user<|end_of_role|>Word: {word}
<|start_of_role|>assistant<|end_of_role|>
Category:"""
        try:
            model_manager.model.reset()
            res = model_manager.model.create_completion(prompt=prompt, max_tokens=20, temperature=0)['choices'][0]['text'].strip().upper()
            
            res_label = "NONE"
            for lbl in ["ORGANIZATION", "LOCATION", "ADDRESS", "PERSON"]:
                if lbl in res:
                    res_label = lbl
                    break
            
            if res_label != "NONE":
                final_entities.append((word, f"{tag_prefix}_{res_label}"))
        except: pass
    return final_entities

def process_pii(text, mask_char="*", minimize_address=False):
    with _processing_lock:
        if not text: return "Empty", "N/A", [], "Idle"
        text = preprocess_text(text)
        main_lang, _ = get_best_language(text)
        nlp = model_manager.load_stanza(main_lang)
        if not nlp: return "Error", "N/A", [], "Stanza Load Failed"
        doc = nlp(text)
        all_candidates = []
        for sent in doc.sentences:
            sent_text = sent.text
            sent_candidates = []
            for m in re.finditer(r'\b\+?[\d\s\-.()]{5,25}\b', sent_text):
                word = m.group().strip()
                if any(c.isdigit() for c in word): sent_candidates.append((word, "NUMERIC_CANDIDATE"))
            for m in EMAIL_REGEX.finditer(sent_text): sent_candidates.append((m.group(), "EMAIL"))
            for m in PCC_REGEX.finditer(sent_text): sent_candidates.append((m.group(), "PCC"))
            for c in scan_address_candidates(sent_text, main_lang) + scan_tracking_numbers(sent_text):
                sent_candidates.append((c["word"], c["type"]))
            if hasattr(sent, 'entities'):
                for ent in sent.entities:
                    if ent.type in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC']: sent_candidates.append((ent.text, "PROPN"))
            for word in sent.words:
                if word.upos in ['PROPN', 'X']: sent_candidates.append((word.text, "PROPN"))
            all_candidates.extend(sent_candidates)
        model_manager.unload_stanza()
        success, msg = model_manager.load_llm()
        if not success: return "Error", "N/A", [], msg
        unique_cands = sorted(list(set(all_candidates)), key=lambda x: len(x[0]), reverse=True)
        verified_cands = []
        high_conf_words = [] 
        for word, p_type in unique_cands:
            if any(word in h_word for h_word in high_conf_words if word != h_word): continue
            if p_type in ["EMAIL", "PCC", "TRACKING_NUMBER", "ZIPCODE", "NUMERIC_CANDIDATE", "ADDRESS"]:
                verified_cands.append((word, p_type))
                high_conf_words.append(word)
            else:
                idx = text.find(word)
                win_text = text[max(0, idx-40):min(len(text), idx+len(word)+40)]
                if verify_with_llm(word, p_type, win_text):
                    verified_cands.append((word, p_type))
                    high_conf_words.append(word)
        refined_llm = refine_entities_with_hint(text, verified_cands, main_lang)
        final_entities = finalize_contextual_labels(text, refined_llm, main_lang)
        unique_entities = sorted(list(set(final_entities)), key=lambda x: len(str(x[0])), reverse=True)
        masked_text = text
        for word, p_type in unique_entities:
            if len(word) < 2 and not any(t in p_type for t in ["PHONE", "EMAIL", "PCC"]): continue
            if minimize_address:
                if any(t in p_type for t in ["STREET", "ZIPCODE", "ADDRESS", "PCC", "TRACKING_NUMBER", "PHONE", "EMAIL"]):
                    masked_text = masked_text.replace(word, mask_char * len(word))
                elif any(word.endswith(s) for s in ["구", "군", "읍", "면", "동", "리", "丁目", "番", "号", "街道", "路", "街"]):
                    masked_text = masked_text.replace(word, mask_char * len(word))
            else:
                masked_text = masked_text.replace(word, mask_char * len(word))
        return masked_text, f"Lang: {main_lang}", unique_entities, "Linguistic PII Detection Complete"
