from model_manager import model_manager
from config import LANG_REGISTRY

def download_all():
    print("--- PII Shield Global: Multi-Language Model Downloader ---")
    
    # 1. LLM 모델 다운로드 시도
    print("\n[1] Checking LLM Model...")
    success, msg = model_manager.load_llm()
    if success:
        print("[LLM] Model is ready.")
        model_manager.unload_llm()
    else:
        print(f"[LLM] Error: {msg}")

    # 2. Stanza 모델 전체 다운로드
    print("\n[2] Downloading Stanza Models for all registered languages...")
    for code, info in LANG_REGISTRY.items():
        stanza_code = 'zh-hans' if code == 'zh' else code
        print(f"\nProcessing {info['name']} ({stanza_code})...")
        success, msg = model_manager.ensure_stanza_download(stanza_code)
        if success:
            print(f"[Stanza] {info['name']}: {msg}")
        else:
            print(f"[Stanza] {info['name']}: {msg}")

    # 3. 최종 상태 보고
    print("\n[3] Final Installation Status Report")
    print("-" * 50)
    print(f"{'Language':<15} | {'Code':<8} | {'Status':<15}")
    print("-" * 50)
    status = model_manager.get_stanza_status()
    for code, info in status.items():
        status_str = "Installed" if info["installed"] else "Missing"
        print(f"{info['name']:<15} | {code:<8} | {status_str:<15}")
    print("-" * 50)

    print("\n--- All downloads completed! ---")

if __name__ == "__main__":
    download_all()
