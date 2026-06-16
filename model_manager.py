import os
import torch
import gc
import stanza
import requests
from llama_cpp import Llama
from config import MODEL_NAME, GGUF_MODEL_PATH, GGUF_MODEL_URL, STANZA_RESOURCES_DIR, LLM_MODELS_DIR

class ModelManager:
    def __init__(self):
        self.model = None
        self.stanza_pipeline = None
        self.current_stanza_lang = None

    def _clear_vram(self):
        """VRAM 캐시 및 가비지 컬렉션 강제 실행"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def download_gguf_model(self):
        """GGUF 모델 다운로드"""
        if os.path.exists(GGUF_MODEL_PATH):
            return True, "Model already exists"
        
        os.makedirs(LLM_MODELS_DIR, exist_ok=True)
        print(f"📥 Downloading GGUF model: {MODEL_NAME}...")
        try:
            response = requests.get(GGUF_MODEL_URL, stream=True)
            response.raise_for_status()
            with open(GGUF_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Download complete!")
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def load_llm(self):
        """llama-cpp-python을 사용하여 GGUF 모델 로드 (Stanza 제거 후)"""
        if self.model is not None:
            return True, "LLM already loaded"
        
        # Stanza 제거
        self.unload_stanza()
        
        # 모델 존재 여부 확인 및 다운로드
        if not os.path.exists(GGUF_MODEL_PATH):
            success, msg = self.download_gguf_model()
            if not success:
                return False, f"Download failed: {msg}"

        print(f"🚀 Loading GGUF model with llama-cpp: {MODEL_NAME}...")
        try:
            # Hybrid Mamba2 MoE 지원을 위한 보수적 설정
            self.model = Llama(
                model_path=GGUF_MODEL_PATH,
                n_ctx=2048,
                n_batch=512,
                n_gpu_layers=-1 if torch.cuda.is_available() else 0,
                verbose=True # 에러 추적을 위해 verbose 켬
            )
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def unload_llm(self):
        """LLM을 VRAM/RAM에서 제거"""
        if self.model is not None:
            print("🗑️ Unloading LLM (llama-cpp)...")
            del self.model
            self.model = None
            self._clear_vram()

    def load_stanza(self, lang, processors='tokenize,pos'):
        """Stanza를 로드 (VRAM 점유 최소화를 위해 CPU 사용)"""
        if self.current_stanza_lang == lang and self.stanza_pipeline is not None:
            return self.stanza_pipeline

        # LLM 제거 후 로드
        self.unload_llm()
        self.unload_stanza()

        print(f"🚀 Loading Stanza ({lang}) with [{processors}] on CPU...")
        try:
            self.stanza_pipeline = stanza.Pipeline(
                lang=lang, 
                model_dir=STANZA_RESOURCES_DIR,
                processors=processors,
                download_method=None,
                use_gpu=False
            )
            self.current_stanza_lang = lang
            return self.stanza_pipeline
        except Exception as e:
            print(f"Stanza load failed: {e}")
            return None

    def unload_stanza(self):
        """Stanza를 제거"""
        if self.stanza_pipeline is not None:
            print(f"🗑️ Unloading Stanza ({self.current_stanza_lang})...")
            del self.stanza_pipeline
            self.stanza_pipeline = None
            self.current_stanza_lang = None
            self._clear_vram()

    def download_stanza_model(self, lang):
        try:
            stanza.download(lang, model_dir=STANZA_RESOURCES_DIR)
            return True, f"Successfully downloaded Stanza '{lang}' model!"
        except Exception as e:
            return False, str(e)

    def is_stanza_installed(self, lang):
        lang_path = os.path.join(STANZA_RESOURCES_DIR, lang)
        return os.path.exists(lang_path)

# 싱글톤 인스턴스
model_manager = ModelManager()
