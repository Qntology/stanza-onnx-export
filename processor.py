import re
import json
import torch
from config import PII_CATEGORIES
from model_manager import model_manager
from utils import get_best_language

# 정규표현식 패턴
PHONE_REGEX = re.compile(r'(\+?\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]?\d{3,4}[- ]?\d{4,6}')
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def extract_pii_with_llm(text, lang_code):
    """IBM Granite를 이용한 PII 추출"""
    if model_manager.model is None:
        return []

    prompt = f"""<|system|>
You are an expert PII (Personally Identifiable Information) detector.
Identify the following categories in the text: {', '.join(PII_CATEGORIES)}.
Return a JSON array of objects with "word" and "type" keys. 
If no PII is found, return [].
<|user|>
Language: {lang_code}
Text: {text}
<|assistant|>
JSON:"""
    
    inputs = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.model.device)
    with torch.no_grad():
        outputs = model_manager.model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False, # Deterministic
            pad_token_id=model_manager.tokenizer.eos_token_id
        )
    
    response = model_manager.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    try:
        # JSON 블록 추출
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
            
        match = re.search(r'\[.*\]', json_str, re.DOTALL)
        if match:
            data = json.loads(match.group())
            # 리스트 형식인지 확인
            return data if isinstance(data, list) else []
        return []
    except:
        return []

def process_pii(text, mask_char="*"):
    if not text:
        return "입력 텍스트가 비어 있습니다.", "N/A", []

    # 1. 언어 감지 (Main/Sub 분석)
    main_lang, sub_langs = get_best_language(text)
    
    # 2. Stanza 파이프라인 가져오기 (Main 언어 기준)
    nlp = model_manager.get_stanza_pipeline(main_lang)
    
    results = []
    
    # 3. Regex 기반 탐지 (Phone, Email)
    for m in PHONE_REGEX.finditer(text):
        results.append({"word": m.group(), "type": "PHONE"})
    for m in EMAIL_REGEX.finditer(text):
        results.append({"word": m.group(), "type": "EMAIL"})
    
    # 4. LLM & Stanza 기반 탐지
    pos_info = f"Main Language: {main_lang}\nSub Languages: {', '.join(sub_langs) if sub_langs else 'None'}\n"
    
    if nlp:
        doc = nlp(text)
        # Stanza NER 결과 추가
        for ent in doc.entities:
            results.append({"word": ent.text, "type": ent.type})
            
        for sent in doc.sentences:
            # LLM 탐지 병행
            llm_found = extract_pii_with_llm(sent.text, main_lang)
            results.extend(llm_found)
            # 형태소 분석 정보 (샘플링)
            pos_info += " / ".join([f"{w.text}({w.upos})" for w in sent.words[:10]]) + "...\n"
    else:
        pos_info += f"⚠️ Stanza model for '{main_lang}' not installed. Using LLM/Regex only.\n"
        results.extend(extract_pii_with_llm(text, main_lang))

    # 5. 결과 정제 (중복 제거 및 마스킹)
    unique_results = []
    seen_words = set()
    for res in sorted(results, key=lambda x: len(x.get('word','')), reverse=True):
        word = res.get('word', '')
        if word and word not in seen_words:
            unique_results.append(res)
            seen_words.add(word)

    masked_text = text
    for res in unique_results:
        word = res['word']
        if word in masked_text:
            if mask_char == "[TAG]":
                masked_text = masked_text.replace(word, f"[{res['type']}]")
            else:
                masked_text = masked_text.replace(word, mask_char * len(word))
            
    return masked_text, pos_info, [(r['word'], r['type']) for r in unique_results]
