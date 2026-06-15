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

def extract_pii_with_llm(text, lang_code, category):
    """특정 카테고리에 대해서만 PII 추출 (프롬프트 간소화 및 범용성 강화)"""
    if model_manager.model is None:
        return []

    prompt = f"""<|start_of_role|>system<|end_of_role|>You are a strict PII detector focusing ONLY on the category: {category}.
Rules:
1. Identify all instances of {category} in the given text.
2. Return ONLY a JSON array of strings. Example: ["found_item1", "found_item2"]
3. If none found, return [].
4. DO NOT explain. DO NOT include other categories.
<|start_of_role|>user<|end_of_role|>Language: {lang_code}
Text: {text}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
JSON:"""
    
    inputs = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.model.device)
    with torch.no_grad():
        outputs = model_manager.model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False,
            pad_token_id=model_manager.tokenizer.eos_token_id
        )
    
    response = model_manager.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    try:
        json_str = response.strip()
        match = re.search(r'\[.*\]', json_str, re.DOTALL)
        if match:
            words = json.loads(match.group())
            if isinstance(words, list):
                return [{"word": str(w), "type": category} for w in words if w]
        return []
    except:
        return []

def process_pii(text, mask_char="*"):
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
    # Stanza는 문장 단위(Sentence)로 텍스트를 쪼개어 맥락을 유지하면서도 최소 단위를 확보합니다.
    nlp = model_manager.load_stanza(main_lang)
    units = []
    pos_info = f"Main Language: {main_lang}\n"
    
    if nlp:
        doc = nlp(text)
        # 문장 단위로 분할
        units = [sent.text for sent in doc.sentences]
        if doc.sentences:
            pos_info += " / ".join([f"{w.text}({w.upos})" for w in doc.sentences[0].words[:15]]) + "...\n"
        print(f"[DEBUG] Stanza split text into {len(units)} sentences. Now unloading Stanza...")
        model_manager.unload_stanza()
    else:
        units = [line.strip() for line in text.split('\n') if line.strip()]
        print(f"[DEBUG] Fallback to line split. {len(units)} units.")

    # --- [VRAM STEP 2] LLM 로드 및 PII 탐지 ---
    success, msg = model_manager.load_llm()
    if not success:
        return "LLM 로드 실패", "N/A", [], f"Error: {msg}"

    results = []
    for i, unit in enumerate(units):
        print(f"  > Processing Unit {i+1}/{len(units)}: {unit[:30]}...")
        # Regex (전역적이지만 단위 내 확인)
        for m in PHONE_REGEX.finditer(unit): results.append({"word": m.group(), "type": "PHONE"})
        for m in EMAIL_REGEX.finditer(unit): results.append({"word": m.group(), "type": "EMAIL"})
        
        # LLM 탐지 - 각 카테고리별로 개별 추론 진행
        for cat in PII_CATEGORIES:
            llm_found = extract_pii_with_llm(unit, main_lang, cat)
            results.extend(llm_found)

    # 4. 결과 정제 및 마스킹 (중복 제거 및 길이순 정렬하여 치환)
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
            
    return masked_text, pos_info, [(r['word'], r['type']) for r in unique_results], download_msg
