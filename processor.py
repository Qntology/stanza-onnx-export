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

def classify_pii_with_pos(word, pos_tag, context, main_lang):
    from config import LANG_REGISTRY
    from utils import get_best_language
    
    # 단어의 지배적 언어 감지
    word_lang, _ = get_best_language(word)
    
    # 레지스트리에서 정보 조회
    lang_info = LANG_REGISTRY.get(word_lang, LANG_REGISTRY.get(main_lang, {"name": "Unknown", "prefix": "UN"}))
    script_name = lang_info["name"]
    tag_prefix = lang_info["prefix"]
    
    tag_map = {
        'PROPN': ["PERSON", "ORGANIZATION", "LOCATION"],
        'NUM': ["AGE", "BANKACCOUNT", "CREDITCARD", "SSN", "DATE"],
        'NOUN': ["JOBTITLE", "OCCUPATION"]
    }
    target_labels = tag_map.get(pos_tag, ["PERSON", "ORGANIZATION"])
    
    # 언어 컨텍스트를 주입한 정밀 프롬프트
    prompt = f"""<|start_of_role|>system<|end_of_role|>Task: PII Classification
Language Context: {script_name}
Context: {context}
Question: In {script_name} context, is the word '{word}' one of these: {", ".join(target_labels)}?
Rules: Return ONLY the LABEL or 'NONE'.
<|start_of_role|>user<|end_of_role|>Word: {word}
<|start_of_role|>assistant<|end_of_role|>
Label:"""
    try:
        model_manager.model.reset()
        res = model_manager.model.create_completion(prompt=prompt, max_tokens=10, temperature=0)['choices'][0]['text'].strip().upper()
        for lbl in target_labels:
            if lbl in res: 
                # [변경] 태그 정규화: KO_PERSON, EN_ORGANIZATION 등
                return f"{tag_prefix}_{lbl}"
        return None
    except: return None

def refine_names_with_hint(text, found_results, main_lang):
    """탐지된 이름을 HINT로 사용하여 주변의 성/이름 누락분을 2차 탐지"""
    from config import LANG_REGISTRY
    lang_info = LANG_REGISTRY.get(main_lang, {"name": "Unknown", "prefix": "UN"})
    tag_prefix = lang_info["prefix"]
    
    person_results = [r for r in found_results if "PERSON" in r[1]]
    new_finds = []
    
    for word, p_type in person_results:
        # 단어 주변 컨텍스트 추출 (앞뒤 15자)
        idx = text.find(word)
        if idx == -1: continue
        context_window = text[max(0, idx-15):min(len(text), idx+len(word)+15)]
        
        # [HINT 기반 프롬프트]
        prompt = f"""<|start_of_role|>system<|end_of_role|>Task: Name Decomposition
HINT: The word '{word}' is already confirmed as a PERSON name in {lang_info['name']}.
Context: ...{context_window}...
Question: Looking at the context, identify the full name including 'First Name' and 'Last Name'.
Rules: Return in JSON format: {{"full_name": "...", "first_name": "...", "last_name": "..."}}
<|start_of_role|>user<|end_of_role|>Context: {context_window}
<|start_of_role|>assistant<|end_of_role|>
{{"full_name":"""
        
        try:
            model_manager.model.reset()
            # "{"full_name":" 부분을 미리 채워넣어 JSON 유도
            res_raw = model_manager.model.create_completion(prompt=prompt, max_tokens=60, temperature=0)['choices'][0]['text'].strip()
            res_json = json.loads('{"full_name":' + res_raw if not res_raw.startswith('{') else res_raw)
            
            # 새로 발견된 부분들을 결과에 추가
            for key in ["full_name", "first_name", "last_name"]:
                val = res_json.get(key, "")
                if val and len(val) > 1 and val in context_window:
                    label = "FIRSTNAME" if "first" in key else "LASTNAME" if "last" in key else "PERSON"
                    new_finds.append((val, f"{tag_prefix}_{label}"))
        except: continue
        
    return list(set(found_results + new_finds))

def process_pii(text, mask_char="*", minimize_address=False):
    with _processing_lock:
        if not text: return "Empty", "N/A", [], "Idle"
        
        # [신규] 전처리: 특수문자 주변 공백 삽입 (단어 분리 최적화)
        text = preprocess_text(text)
        
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
            label = classify_pii_with_pos(word, cand["upos"], cand["context"], main_lang)
            if label: final_results.append((word, label))

        # [신규] 2단계 HINT 기반 성/이름 정밀 추론
        final_results = refine_names_with_hint(text, final_results, main_lang)

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
