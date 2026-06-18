import os
import torch
import gc
import stanza
import requests
import sys
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
        print(f"[Model] Downloading GGUF model: {MODEL_NAME}...", file=sys.stderr)
        try:
            response = requests.get(GGUF_MODEL_URL, stream=True)
            response.raise_for_status()
            with open(GGUF_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("[Model] Download complete!", file=sys.stderr)
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

        print(f"[Model] Loading GGUF model with llama-cpp: {MODEL_NAME}...", file=sys.stderr)
        try:
            # Hybrid Mamba2 MoE 지원을 위한 보수적 설정
            self.model = Llama(
                model_path=GGUF_MODEL_PATH,
                n_ctx=2048,
                n_batch=512,
                n_gpu_layers=-1 if torch.cuda.is_available() else 0,
                verbose=False
            )
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def unload_llm(self):
        """LLM을 VRAM/RAM에서 제거"""
        if self.model is not None:
            print("[Model] Unloading LLM (llama-cpp)...", file=sys.stderr)
            del self.model
            self.model = None
            self._clear_vram()

    def ensure_stanza_download(self, lang):
        """해당 언어의 Stanza 모델이 없으면 다운로드"""
        if not self.is_stanza_installed(lang):
            print(f"[Stanza] model for '{lang}' not found. Downloading...", file=sys.stderr)
            try:
                # resources.json이 없으면 전체 체크가 안되므로 download 실행
                stanza.download(lang, model_dir=STANZA_RESOURCES_DIR, verbose=False)
                return True, "Download Complete"
            except Exception as e:
                return False, str(e)
        return True, "Already Installed"

    def load_stanza(self, lang, processors=None):
        """Stanza를 로드 (VRAM 점유 최소화를 위해 CPU 사용)"""
        from config import LANG_REGISTRY
        
        # 언어 코드 정규화 (Stanza 호환)
        stanza_lang = lang
        if lang == 'zh': stanza_lang = 'zh-hans' # 기본값
        
        if self.current_stanza_lang == stanza_lang and self.stanza_pipeline is not None:
            return self.stanza_pipeline

        # LLM 제거 후 로드
        self.unload_llm()
        self.unload_stanza()

        # 레지스트리에서 권장 프로세서 가져오기
        lang_info = LANG_REGISTRY.get(lang, {})
        if processors is None:
            processors = lang_info.get("processors", "tokenize,pos")

        # 다운로드 확인
        self.ensure_stanza_download(stanza_lang)

        print(f"[Stanza] Loading ({stanza_lang}) with [{processors}] on CPU...", file=sys.stderr)
        try:
            # 유동적 프로세서 대응: 특정 프로세서 로드 실패 시 단계적 축소 시도
            try:
                self.stanza_pipeline = stanza.Pipeline(
                    lang=stanza_lang, 
                    model_dir=STANZA_RESOURCES_DIR,
                    processors=processors,
                    use_gpu=False,
                    logging_level='ERROR'
                )
            except Exception as e:
                if 'mwt' in processors:
                    print(f"[Stanza] Warning: MWT processor not supported for {stanza_lang}. Retrying without it...", file=sys.stderr)
                    processors = processors.replace(',mwt', '').replace('mwt,', '')
                    self.stanza_pipeline = stanza.Pipeline(
                        lang=stanza_lang, 
                        model_dir=STANZA_RESOURCES_DIR,
                        processors=processors,
                        use_gpu=False,
                        logging_level='ERROR'
                    )
                else:
                    raise e

            self.current_stanza_lang = stanza_lang
            return self.stanza_pipeline
        except Exception as e:
            print(f"[Stanza] Error: load failed for {stanza_lang}: {e}", file=sys.stderr)
            return None

    def unload_stanza(self):
        """Stanza를 제거"""
        if self.stanza_pipeline is not None:
            print(f"[Stanza] Unloading ({self.current_stanza_lang})...", file=sys.stderr)
            del self.stanza_pipeline
            self.stanza_pipeline = None
            self.current_stanza_lang = None
            self._clear_vram()

    def download_stanza_model(self, lang):
        """수동 다운로드용 (UI 등에서 호출)"""
        try:
            stanza.download(lang, model_dir=STANZA_RESOURCES_DIR)
            return True, f"Successfully downloaded Stanza '{lang}' model!"
        except Exception as e:
            return False, str(e)

    def is_stanza_installed(self, lang):
        """단순 폴더 존재 확인 이상의 검증 로직"""
        lang_path = os.path.join(STANZA_RESOURCES_DIR, lang)
        # 1. 폴더 존재 확인
        if not os.path.exists(lang_path):
            return False
        
        # 2. 필수 리소스 파일 확인 (간이 검사)
        files = os.listdir(lang_path)
        if len(files) < 2: 
            return False
            
        return True

    def get_stanza_status(self):
        """전체 모델 설치 상태 요약 반환"""
        from config import LANG_REGISTRY
        status = {}
        for code, info in LANG_REGISTRY.items():
            stanza_code = 'zh-hans' if code == 'zh' else code
            installed = self.is_stanza_installed(stanza_code)
            status[code] = {
                "name": info["name"],
                "installed": installed,
                "path": os.path.join(STANZA_RESOURCES_DIR, stanza_code) if installed else None
            }
        return status

    def get_installed_stanza_models(self):
        """설치된 언어 코드 목록 반환"""
        from config import SUPPORTED_LANGUAGES
        installed = []
        for lang in SUPPORTED_LANGUAGES.keys():
            stanza_lang = 'zh-hans' if lang == 'zh' else lang
            if self.is_stanza_installed(stanza_lang):
                installed.append(lang)
        return installed

# 싱글톤 인스턴스
model_manager = ModelManager()
