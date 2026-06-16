import re
import json
from model_manager import model_manager
from utils import get_best_language
from config import PII_CATEGORIES_MAP

# 정규식 - 강력한 패턴은 무조건 정규식으로
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{3,4}[- ]?\d{4}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def classify_pii_with_pos(word, pos_tag, context):
    """POS 태그를 기반으로 후보 단어의 PII 여부를 정밀 분류 (llama-cpp-python 기반)"""
    if model_manager.model is None: return None
    
    # POS 태그별 매핑될 수 있는 논리적 라벨 그룹 정의
    tag_map = {
        'PROPN': ["FIRSTNAME", "LASTNAME", "CITY", "STATE", "ORGANIZATION", "STREET", "ADDRESS"],
        'NUM': ["AGE", "ZIPCODE", "PHONE", "BANKACCOUNT", "CREDITCARD", "AMOUNT", "SSN", "DATE", "STREET", "BUILDINGNUMBER"],
        'NOUN': ["JOBTITLE", "OCCUPATION", "STREET", "BUILDINGNUMBER", "ADDRESS"]
    }
    
    target_labels = tag_map.get(pos_tag, ["FIRSTNAME", "CITY", "STREET", "ORGANIZATION"])
    labels_str = ", ".join(target_labels)

    # Granite GGUF 프롬프트 구성
    prompt = f"""<|start_of_role|>system<|end_of_role|>Is the word '{word}' a PII in the given context?
Context: {context}
Target Labels: {labels_str}

Rules:
1. Return ONLY the LABEL NAME if it matches perfectly.
2. Return 'NONE' if it is a common word, header, or general noun.
3. Example: 'Notice' -> NONE, 'Seoul' -> CITY.
<|start_of_role|>user<|end_of_role|>Word: {word}
<|start_of_role|>assistant<|end_of_role|>
Label:"""
    
    try:
        # KV 캐시 충돌 방지를 위해 컨텍스트 리셋
        model_manager.model.reset()
        
        response_data = model_manager.model.create_completion(
            prompt=prompt,
            max_tokens=10,
            stop=["<|end_of_text|>", "\n"],
            temperature=0,
            echo=False
        )
        
        response = response_data['choices'][0]['text'].strip().upper()
        
        if "NONE" in response: return None
        for lbl in target_labels:
            if lbl in response: return lbl
        return None
    except Exception as e:
        print(f"LLM Classification failed: {e}")
        return None

def process_pii(text, mask_char="*", minimize_address=False):
    if not text: return "Empty input", "N/A", [], "Idle"
    main_lang, _ = get_best_language(text)
    
    # [1] Stanza 분석 (PII 후보군 추출)
    nlp = model_manager.load_stanza(main_lang, processors='tokenize,pos')
    candidates = []
    if nlp:
        doc = nlp(text)
        for sent in doc.sentences:
            for word in sent.words:
                # 고유명사, 숫자, 기타 외래어(X)를 후보로 선정
                if word.upos in ['PROPN', 'NUM', 'X'] or (word.xpos == 'nq' and len(word.text) > 1):
                    candidates.append({"word": word.text, "upos": word.upos, "context": sent.text})
        model_manager.unload_stanza()
    
    # [2] LLM 로드 (llama-cpp)
    success, msg = model_manager.load_llm()
    if not success: return "LLM Load Failed", "N/A", [], f"Error: {msg}"
    
    final_entities = []
    # 정규식 탐지 (가장 정확)
    for m in PHONE_REGEX.finditer(text): final_entities.append((m.group(), "PHONE"))
    for m in EMAIL_REGEX.finditer(text): final_entities.append((m.group(), "EMAIL"))

    # [3] POS-Guided 정밀 분류
    unique_cand = []
    seen = set()
    for c in candidates:
        if c["word"] not in seen:
            unique_cand.append(c)
            seen.add(c["word"])

    print(f"Analyzing {len(unique_cand)} linguistic candidates with Granite GGUF...")
    for i, c in enumerate(unique_cand):
        word, pos = c["word"], c["upos"]
        # 이미 정규식에 걸린 단어는 제외
        if any(word in str(e[0]) for e in final_entities): continue
        
        print(f"[{i+1}/{len(unique_cand)}] '{word}' ({pos})...")
        label = classify_pii_with_pos(word, pos, c["context"])
        if label:
            print(f"   -> Confirmed: {label}")
            final_entities.append((word, label))

    # [4] 마스킹 (긴 단어부터)
    unique_entities = sorted(list(set(final_entities)), key=lambda x: len(str(x[0])), reverse=True)
    masked_text = text
    address_mask_targets = ["STREET", "ZIPCODE", "BUILDINGNUMBER", "ADDRESS", "SECONDARYADDRESS"]
    address_expose_targets = ["CITY", "STATE", "COUNTY"]

    for word, p_type in unique_entities:
        if minimize_address and p_type in address_mask_targets:
            masked_text = masked_text.replace(word, mask_char * len(word))
        elif minimize_address and p_type in address_expose_targets: pass
        else:
            masked_text = masked_text.replace(word, mask_char * len(word))
            
    return masked_text, f"Lang: {main_lang}", unique_entities, "✨ POS-Guided Hybrid (GGUF) Complete"
