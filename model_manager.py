import os
import torch
import gc
import stanza
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, STANZA_RESOURCES_DIR

class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.stanza_pipeline = None
        self.current_stanza_lang = None

    def _clear_vram(self):
        """VRAM 캐시 및 가비지 컬렉션 강제 실행"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_llm(self):
        """IBM Granite 모델을 VRAM에 로드 (Stanza가 있다면 먼저 제거)"""
        if self.model is not None:
            return True, "LLM already loaded"
        
        # Stanza 제거 후 로드
        self.unload_stanza()
        
        print(f"🚀 Loading LLM to VRAM: {MODEL_NAME}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16, # FP16으로 메모리 절반 감소
                device_map="auto",
                trust_remote_code=True
            )
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def unload_llm(self):
        """LLM을 VRAM에서 제거"""
        if self.model is not None:
            print("🗑️ Unloading LLM from VRAM...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._clear_vram()

    def load_stanza(self, lang):
        """Stanza를 로드 (VRAM 점유 최소화를 위해 CPU 사용)"""
        if self.current_stanza_lang == lang and self.stanza_pipeline is not None:
            return self.stanza_pipeline

        # LLM 제거 후 로드
        self.unload_llm()
        self.unload_stanza()

        print(f"🚀 Loading Stanza ({lang}) on CPU to save VRAM...")
        try:
            self.stanza_pipeline = stanza.Pipeline(
                lang=lang, 
                model_dir=STANZA_RESOURCES_DIR,
                processors='tokenize', # pos 제거
                download_method=None,
                use_gpu=False # CPU 사용
            )
            self.current_stanza_lang = lang
            return self.stanza_pipeline
        except Exception as e:
            print(f"Stanza load failed: {e}")
            return None

    def unload_stanza(self):
        """Stanza를 VRAM에서 제거"""
        if self.stanza_pipeline is not None:
            print(f"🗑️ Unloading Stanza ({self.current_stanza_lang}) from VRAM...")
            del self.stanza_pipeline
            self.stanza_pipeline = None
            self.current_stanza_lang = None
            self._clear_vram()

    def download_stanza_model(self, lang):
        """Stanza 모델 다운로드 (경로 유지)"""
        try:
            stanza.download(lang, model_dir=STANZA_RESOURCES_DIR)
            return True, f"Successfully downloaded Stanza '{lang}' model!"
        except Exception as e:
            return False, str(e)

    def is_stanza_installed(self, lang):
        lang_path = os.path.join(STANZA_RESOURCES_DIR, lang)
        return os.path.exists(lang_path)

    def get_installed_stanza_models(self):
        if not os.path.exists(STANZA_RESOURCES_DIR):
            return []
        return [d for d in os.listdir(STANZA_RESOURCES_DIR) 
                if os.path.isdir(os.path.join(STANZA_RESOURCES_DIR, d))]

# 싱글톤 인스턴스
model_manager = ModelManager()
