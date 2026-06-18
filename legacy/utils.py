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
        "ko": 0,      # 한글
        "zh-hans": 0, # 한자 (중국어 간체)
        "ja": 0,      # 일본어 (히라가나/가타카나)
        "en": 0,      # 라틴 (영어/유럽어)
        "ar": 0,      # 아랍어
        "cs": 0,      # 체코어 (라틴 확장이지만 별도 처리 필요시)
        "other": 0
    }

    for char in text:
        cp = ord(char)
        # 한글: 0xAC00 ~ 0xD7A3, 0x1100 ~ 0x11FF (자음/모음)
        if (0xAC00 <= cp <= 0xD7A3) or (0x1100 <= cp <= 0x11FF):
            scores["ko"] += 2 # 한글은 특징이 강하므로 가중치
        # 일본어 (히라가나/가타카나): 0x3040 ~ 0x30FF
        elif 0x3040 <= cp <= 0x30FF:
            scores["ja"] += 2 # 일본어 전용 문자 가중치
        # 한자 (CJK Unified Ideographs): 0x4E00 ~ 0x9FFF
        elif 0x4E00 <= cp <= 0x9FFF:
            # 일본어 전용 문자가 이미 발견되었다면 ja로 가산, 아니면 zh
            if scores["ja"] > 0:
                scores["ja"] += 1
            else:
                scores["zh-hans"] += 1
        # 아랍어: 0x0600 ~ 0x06FF
        elif 0x0600 <= cp <= 0x06FF:
            scores["ar"] += 2
        # 영어/라틴 (기본 + 확장): 0x0041~0x005A, 0x0061~0x007A, 0x00C0~0x00FF
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x00FF):
            scores["en"] += 1
        else:
            if not char.isspace() and not char.isdigit() and char not in '.,!?()[]{}<>:;@#$%^&*-=_+|/\\\'"':
                scores["other"] += 1

    # 점수 순으로 정렬 (ja/zh-hans 재검증: ja 점수가 있고 zh-hans 점수가 있으면 합산하여 ja로 판단할지 결정)
    if scores["ja"] > 0 and scores["zh-hans"] > 0:
        # 일본어 텍스트에 섞인 한자 처리
        scores["ja"] += scores["zh-hans"]
        scores["zh-hans"] = 0

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
        # langdetect는 문장이 길면 신뢰도가 높지만 짧으면 유니코드가 더 정확함
        ld_lang = detect(text)
        # langdetect 결과를 보조로 사용
        if ld_lang == 'zh-cn': ld_lang = 'zh-hans'
        if ld_lang == 'zh-tw': ld_lang = 'zh-hant'
    except:
        ld_lang = "en"

    # 유니코드 분석 결과를 최우선으로 함
    final_main = analysis["main"]
    
    # 예외 처리: 유니코드가 'en'인데 langdetect가 명확히 다른 유럽어(es, fr, de 등)를 찍으면 변경
    if final_main == "en" and ld_lang in ["es", "fr", "de", "it", "pt", "nl", "cs", "ar"]:
        final_main = ld_lang

    return final_main, analysis["sub"]
