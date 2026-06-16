import os

# 1. 언어 레지스트리 (Registry)
LANG_REGISTRY = {
    "ko": {"name": "Korean", "prefix": "KO", "scripts": ["Hangul"], "processors": "tokenize,pos,lemma,depparse"},
    "en": {"name": "English", "prefix": "EN", "scripts": ["Latin"], "processors": "tokenize,mwt,pos,lemma,depparse"},
    "de": {"name": "German", "prefix": "DE", "scripts": ["Latin"], "processors": "tokenize,mwt,pos,lemma,depparse"},
    "es": {"name": "Spanish", "prefix": "ES", "scripts": ["Latin"], "processors": "tokenize,mwt,pos,lemma,depparse"},
    "fr": {"name": "French", "prefix": "FR", "scripts": ["Latin"], "processors": "tokenize,mwt,pos,lemma,depparse"},
    "ja": {"name": "Japanese", "prefix": "JA", "scripts": ["Hiragana", "Katakana", "Kanji"], "processors": "tokenize,pos,lemma,depparse"},
    "pt": {"name": "Portuguese", "prefix": "PT", "scripts": ["Latin"], "processors": "tokenize,mwt,pos,lemma,depparse"},
    "ar": {"name": "Arabic", "prefix": "AR", "scripts": ["Arabic"], "processors": "tokenize,mwt,pos,lemma,depparse"},
    "cs": {"name": "Czech", "prefix": "CS", "scripts": ["Latin"], "processors": "tokenize,pos,lemma,depparse"},
    "it": {"name": "Italian", "prefix": "IT", "scripts": ["Latin"], "processors": "tokenize,mwt,pos,lemma,depparse"},
    "nl": {"name": "Dutch", "prefix": "NL", "scripts": ["Latin"], "processors": "tokenize,mwt,pos,lemma,depparse"},
    "zh-hans": {"name": "Chinese", "prefix": "ZH", "scripts": ["Han", "Kanji"], "processors": "tokenize,pos,lemma,depparse,ner"},
}

# 2. Stanza 설정
STANZA_RESOURCES_DIR = os.path.join(os.getenv('APPDATA'), 'xtanza')
SUPPORTED_LANGUAGES = {k: v["name"] for k, v in LANG_REGISTRY.items()}
