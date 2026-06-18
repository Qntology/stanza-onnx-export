import re

# 국가별 전화번호 정규표현식 맵
PHONE_REGEX_MAP = {
    "ko": re.compile(r'(\+82|0)[1-9]\d{0,2}[-.\s]?\d{3,4}[-.\s]?\d{4}'), 
    "en": re.compile(r'(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|(\+44[-.\s]?)?\(?0?\d{2,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}'),
    "ja": re.compile(r'(\+81|0)\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{4}'),
    "zh-hans": re.compile(r'(\+?86[-.\s]?)?1[3-9]\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{3,4}|(\+?86[-.\s]?)?(\d{2,4}[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}'), 
    "fr": re.compile(r'(\+33|0)[1-9]([-.\s]?\d{2}){4}'),
    "es": re.compile(r'(\+34|0)?[6789]\d{2}([-.\s]?\d{3}){2}'),
    "de": re.compile(r'(\+49|0)[1-9]\d{1,4}([-.\s]?\d{3,9})'),
    "pt": re.compile(r'(\+351|0)[29]\d{8}|(\+55|0)?\s?(\(?\d{2}\)?)\s?9?\d{4}-?\d{4}'),
    "ar": re.compile(r'(\+966|0)?5\d{8}|(\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}'),
    "cs": re.compile(r'(\+420)?\s?\d{3}\s?\d{3}\s?\d{3}'),
    "it": re.compile(r'(\+39)?\s?3\d{2}\s?\d{6,7}'),
    "nl": re.compile(r'(\+31|0)6\d{8}'),
    "default": re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}')
}

# 공통 PII 정규표현식
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
PCC_REGEX = re.compile(r'\bP\d{12}\b', re.IGNORECASE)
TRACKING_CANDIDATE_REGEX = re.compile(r'\b\d{9,14}\b')
NUMBER_BLOCK_REGEX = re.compile(r'^[\d\-\./]+$')
