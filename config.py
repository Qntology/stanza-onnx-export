import os

# 1. LLM 모델 설정
# 사용자가 지정한 IBM Granite 350M 모델
MODEL_NAME = "ibm-granite/granite-4.0-h-350m"
HF_MODEL_URL = f"https://huggingface.co/{MODEL_NAME}"

# 2. Stanza 설정
# 사용자가 요청한 AppData/Roaming/xtanza 경로로 변경
STANZA_RESOURCES_DIR = os.path.join(os.getenv('APPDATA'), 'xtanza')

# 지원하는 주요 언어 코드 (Stanza 기준)
# https://huggingface.co/stanfordnlp/models 참고
SUPPORTED_LANGUAGES = {
    "ko": "Korean",
    "en": "English",
    "zh-hans": "Chinese (Simplified)",
    "ja": "Japanese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "vi": "Vietnamese",
    "th": "Thai"
}

# 3. PII 카테고리
PII_CATEGORIES = ["NAME", "ADDRESS", "ID_CARD", "BANK_ACCOUNT", "PHONE", "EMAIL"]
