import os
from config_languages import LANG_REGISTRY, SUPPORTED_LANGUAGES, STANZA_RESOURCES_DIR
from config_addresses import ADDRESS_DICTIONARY
from config_categories import PII_CATEGORIES_MAP
from config_patterns import PHONE_REGEX_MAP, EMAIL_REGEX, PCC_REGEX, TRACKING_CANDIDATE_REGEX, NUMBER_BLOCK_REGEX

# 1. LLM 모델 설정 (GGUF 기반)
MODEL_NAME = "granite-4.0-h-350m-Q8_0.gguf"
GGUF_MODEL_URL = "https://huggingface.co/ibm-granite/granite-4.0-h-350m-GGUF/resolve/main/granite-4.0-h-350m-Q8_0.gguf"
LLM_MODELS_DIR = os.path.join(os.getenv('APPDATA'), 'pii-models')
GGUF_MODEL_PATH = os.path.join(LLM_MODELS_DIR, MODEL_NAME)

# 2. Stanza 설정 (Stanza 관련 변수는 config_languages에서 가져옴)
# (STANZA_RESOURCES_DIR, SUPPORTED_LANGUAGES)

# 3. 다국어 주소 키워드 및 패턴 사전 (config_addresses에서 가져옴)
# (ADDRESS_DICTIONARY)

# 4. PII 카테고리 (config_categories에서 가져옴)
# (PII_CATEGORIES_MAP)

# 5. 정규표현식 맵 및 공통 정규표현식 (config_patterns에서 가져옴)
# (PHONE_REGEX_MAP, EMAIL_REGEX, PCC_REGEX, TRACKING_CANDIDATE_REGEX, NUMBER_BLOCK_REGEX)
