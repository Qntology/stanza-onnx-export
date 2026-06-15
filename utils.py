import re
from langdetect import detect, DetectorFactory
from config import SUPPORTED_LANGUAGES

# 언어 감지 일관성 유지
DetectorFactory.seed = 0

def analyze_unicode_script(text):
    """
    유니코드 범위를 분석하여 텍스트 내 포함된 언어들의 비중을 계산합니다.
    Main 언어와 Sub 언어를 구분하여 반환합니다.
    """
    if not text:
        return {"main": "en", "sub": []}

    scores = {
        "ko": 0,    # 한글
        "zh-hans": 0, # 한자 (중국어 간체 중심)
        "ja": 0,    # 일본어 (히라가나/가타카나)
        "en": 0,    # 라틴 (영어/유럽어)
        "other": 0
    }

    for char in text:
        cp = ord(char)
        # 한글: 0xAC00 ~ 0xD7A3
        if 0xAC00 <= cp <= 0xD7A3:
            scores["ko"] += 1
        # 한자 (CJK Unified Ideographs): 0x4E00 ~ 0x9FFF
        elif 0x4E00 <= cp <= 0x9FFF:
            # 일본어 텍스트에도 한자가 섞이므로, 나중에 ja 점수와 결합 로직 필요할 수 있음
            scores["zh-hans"] += 1
        # 일본어 (히라가나/가타카나): 0x3040 ~ 0x30FF
        elif 0x3040 <= cp <= 0x30FF:
            scores["ja"] += 1
        # 영어/라틴: 0x0041~0x005A, 0x0061~0x007A
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            scores["en"] += 1
        else:
            scores["other"] += 1

    # 점수 순으로 정렬
    sorted_langs = sorted(
        [(k, v) for k, v in scores.items() if v > 0 and k != "other"],
        key=lambda x: x[1],
        reverse=True
    )

    if not sorted_langs:
        return {"main": "en", "sub": []}

    main_lang = sorted_langs[0][0]
    sub_langs = [lang for lang, score in sorted_langs[1:]]

    return {"main": main_lang, "sub": sub_langs}

def get_best_language(text):
    """
    유니코드 분석과 langdetect를 결합하여 최적의 Stanza 언어 코드를 결정합니다.
    """
    analysis = analyze_unicode_script(text)
    
    try:
        ld_lang = detect(text)
    except:
        ld_lang = "en"

    # Stanza 코드 매핑 보정
    mapping = {"ko": "ko", "en": "en", "zh": "zh-hans", "ja": "ja", "fr": "fr", "de": "de"}
    
    # 유니코드 분석 결과를 우선순위에 둠 (특히 혼합 언어의 경우)
    final_main = mapping.get(analysis["main"], analysis["main"])
    
    # langdetect 결과가 SUPPORTED_LANGUAGES에 있고 유니코드 분석과 다를 경우 sub에 추가 검토 가능
    return final_main, analysis["sub"]
