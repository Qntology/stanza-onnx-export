import os

# 1. LLM 모델 설정 (GGUF 기반)
MODEL_NAME = "granite-4.0-h-350m-Q8_0.gguf"
GGUF_MODEL_URL = "https://huggingface.co/ibm-granite/granite-4.0-h-350m-GGUF/resolve/main/granite-4.0-h-350m-Q8_0.gguf"
# 모델 저장 경로 (AppData/Roaming/pii-models)
LLM_MODELS_DIR = os.path.join(os.getenv('APPDATA'), 'pii-models')
GGUF_MODEL_PATH = os.path.join(LLM_MODELS_DIR, MODEL_NAME)

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

# 3. PII 카테고리 (Full Label Space - 54 categories)
PII_CATEGORIES_MAP = {
    "Identity": ["FIRSTNAME", "MIDDLENAME", "LASTNAME", "PREFIX", "AGE", "GENDER", "SEX", "EYECOLOR", "HEIGHT", "USERNAME", "OCCUPATION", "JOBTITLE", "JOBDEPARTMENT", "ORGANIZATION", "USERAGENT"],
    "Contact": ["EMAIL", "PHONE", "URL"],
    "Address": ["STREET", "BUILDINGNUMBER", "SECONDARYADDRESS", "CITY", "COUNTY", "STATE", "ZIPCODE", "GPSCOORDINATES", "ORDINALDIRECTION"],
    "Dates & time": ["DATE", "DATEOFBIRTH", "TIME"],
    "Government IDs": ["SSN"],
    "Financial": ["ACCOUNTNAME", "BANKACCOUNT", "IBAN", "BIC", "CREDITCARD", "CREDITCARDISSUER", "CVV", "PIN", "MASKEDNUMBER", "AMOUNT", "CURRENCY", "CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL"],
    "Crypto": ["BITCOINADDRESS", "ETHEREUMADDRESS", "LITECOINADDRESS"],
    "Vehicle": ["VIN", "VRM"],
    "Digital": ["IPADDRESS", "MACADDRESS", "IMEI"],
    "Auth": ["PASSWORD"]
}

# 플랫 리스트 (탐지 엔진용)
PII_CATEGORIES = [item for sublist in PII_CATEGORIES_MAP.values() for item in sublist]
