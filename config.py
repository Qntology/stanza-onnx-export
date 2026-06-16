import os

# 1. LLM 모델 설정 (GGUF 기반)
MODEL_NAME = "granite-4.0-h-350m-Q8_0.gguf"
GGUF_MODEL_URL = "https://huggingface.co/ibm-granite/granite-4.0-h-350m-GGUF/resolve/main/granite-4.0-h-350m-Q8_0.gguf"
LLM_MODELS_DIR = os.path.join(os.getenv('APPDATA'), 'pii-models')
GGUF_MODEL_PATH = os.path.join(LLM_MODELS_DIR, MODEL_NAME)

# 2. Stanza 설정
STANZA_RESOURCES_DIR = os.path.join(os.getenv('APPDATA'), 'xtanza')

# 지원하는 주요 언어 코드 (Stanza 기준)
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

# 3. 다국어 주소 키워드 및 패턴 사전
# 'Address-First' 전략을 위해 admin(행정구역)과 street(도로명) 키워드 세분화
ADDRESS_DICTIONARY = {
    "ko": {
        "admin": ["도", "시", "군", "구", "읍", "면", "동", "리"],
        "street": ["대로", "로", "길", "번길"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "large_to_small"
    },
    "en": {
        "admin": ["State", "City", "County", "Province", "Region"],
        "street": ["Street", "St", "Road", "Rd", "Avenue", "Ave", "Lane", "Ln", "Drive", "Dr", "Boulevard", "Bd", "Court", "Ct", "Parkway", "Pkwy"],
        "zipcode_regex": r"\b\d{5}(-\d{4})?\b",
        "order": "small_to_large"
    },
    "de": {
        "admin": ["Bundesland", "Kreis", "Stadt", "Bezirk", "Gemeinde"],
        "street": ["straße", "str.", "weg", "platz", "gasse", "allee", "ring", "damm"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "small_to_large"
    },
    "es": {
        "admin": ["Provincia", "Municipio", "Barrio", "Localidad", "Comunidad", "Ciudad"],
        "street": ["Calle", "C/", "Avenida", "Av.", "Paseo", "Plaza", "Camino", "Ronda", "Estrada"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "small_to_large"
    },
    "fr": {
        "admin": ["Région", "Département", "Arrondissement", "Commune", "Quartier", "Ville"],
        "street": ["Rue", "Avenue", "Av.", "Boulevard", "Bd", "Place", "Chemin", "Route", "Quai", "Allée"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "small_to_large"
    },
    "ja": {
        "admin": ["都", "道", "府", "県", "市", "区", "郡"],
        "street": ["丁目", "番", "号", "町", "村"],
        "zipcode_regex": r"\b\d{3}-\d{4}\b",
        "order": "large_to_small"
    },
    "pt": {
        "admin": ["Distrito", "Estado", "Concelho", "Freguesia", "Bairro", "Cidade"],
        "street": ["Rua", "R.", "Avenida", "Av.", "Praça", "Pç.", "Largo", "Estrada", "Travessa"],
        "zipcode_regex": r"\b\d{4}-\d{3}\b",
        "order": "small_to_large"
    },
    "ar": {
        "admin": ["محافظة", "ولاية", "منطقة", "بلدية", "حي", "مدينة"],
        "street": ["شارع", "طريق", "ساحة", "حارة"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "small_to_large"
    },
    "cs": {
        "admin": ["Kraj", "Okres", "Obec", "Část", "Město"],
        "street": ["Ulice", "ul.", "Náměstí", "nám.", "Třída", "tř.", "Alej", "Nábřeží"],
        "zipcode_regex": r"\b\d{3} \d{2}\b",
        "order": "small_to_large"
    },
    "it": {
        "admin": ["Regione", "Provincia", "Comune", "Frazione", "Città"],
        "street": ["Via", "Viale", "Piazza", "P.za", "Corso", "Largo", "Strada", "Vicolo"],
        "zipcode_regex": r"\b\d{5}\b",
        "order": "small_to_large"
    },
    "nl": {
        "admin": ["Provincie", "Gemeente", "Wijk", "Buurt", "Stad"],
        "street": ["straat", "weg", "plein", "gracht", "singel", "laan", "dijk"],
        "zipcode_regex": r"\b[1-9]\d{3}\s?[A-Z]{2}\b",
        "order": "small_to_large"
    },
    "zh-hans": {
        "admin": ["省", "市", "区", "县", "镇", "乡"],
        "street": ["路", "街", "大道", "巷", "弄"],
        "zipcode_regex": r"\b\d{6}\b",
        "order": "large_to_small"
    }
}

# 4. PII 카테고리 (Full Label Space - 54 categories)
PII_CATEGORIES_MAP = {
    "Identity": ["FIRSTNAME", "MIDDLENAME", "LASTNAME", "PREFIX", "AGE", "GENDER", "SEX", "EYECOLOR", "HEIGHT", "USERNAME", "OCCUPATION", "JOBTITLE", "JOBDEPARTMENT", "ORGANIZATION", "USERAGENT"],
    "Contact": ["EMAIL", "PHONE", "URL"],
    "Address": ["STREET", "BUILDINGNUMBER", "SECONDARYADDRESS", "CITY", "COUNTY", "STATE", "ZIPCODE", "GPSCOORDINATES", "ORDINALDIRECTION"],
    "Dates & time": ["DATE", "DATEOFBIRTH", "TIME"],
    "Government IDs": ["SSN", "PCC"],
    "Financial": ["ACCOUNTNAME", "BANKACCOUNT", "IBAN", "BIC", "CREDITCARD", "CREDITCARDISSUER", "CVV", "PIN", "MASKEDNUMBER", "AMOUNT", "CURRENCY", "CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL", "TRACKING_NUMBER"],
    "Crypto": ["BITCOINADDRESS", "ETHEREUMADDRESS", "LITECOINADDRESS"],
    "Vehicle": ["VIN", "VRM"],
    "Digital": ["IPADDRESS", "MACADDRESS", "IMEI"],
    "Auth": ["PASSWORD"]
}

PII_CATEGORIES = [item for sublist in PII_CATEGORIES_MAP.values() for item in sublist]
