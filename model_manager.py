import os
import torch
import stanza
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, STANZA_RESOURCES_DIR

class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.stanza_pipelines = {}

    def load_llm(self):
        """IBM Granite 모델 로드"""
        print(f"Loading LLM: {MODEL_NAME}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def get_stanza_pipeline(self, lang):
        """Stanza 파이프라인 로드 및 캐싱"""
        if lang not in self.stanza_pipelines:
            try:
                # 설치 여부 확인 후 로드
                self.stanza_pipelines[lang] = stanza.Pipeline(
                    lang=lang, 
                    model_dir=STANZA_RESOURCES_DIR,
                    processors='tokenize,pos,ner', # NER 추가
                    download_method=None
                )
            except:
                return None
        return self.stanza_pipelines[lang]

    def download_stanza_model(self, lang):
        """Stanza 모델 다운로드"""
        try:
            stanza.download(lang, model_dir=STANZA_RESOURCES_DIR)
            return True, f"Successfully downloaded Stanza '{lang}' model!"
        except Exception as e:
            return False, str(e)

    def is_stanza_installed(self, lang):
        """특정 언어 모델 설치 여부 확인"""
        lang_path = os.path.join(STANZA_RESOURCES_DIR, lang)
        return os.path.exists(lang_path)

    def get_installed_stanza_models(self):
        """설치된 모든 Stanza 모델 목록 반환"""
        if not os.path.exists(STANZA_RESOURCES_DIR):
            return []
        return [d for d in os.listdir(STANZA_RESOURCES_DIR) 
                if os.path.isdir(os.path.join(STANZA_RESOURCES_DIR, d))]

# 싱글톤 인스턴스
model_manager = ModelManager()
