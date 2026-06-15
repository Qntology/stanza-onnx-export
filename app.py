import gradio as gr
from config import MODEL_NAME, HF_MODEL_URL, SUPPORTED_LANGUAGES
from model_manager import model_manager
from processor import process_pii

# 초기화 시 모델 로드 시도
print("Initializing Models...")
success, msg = model_manager.load_llm()
if not success:
    print(f"Initial LLM Load Failed: {msg}")

def handle_stanza_download(lang_code):
    if not lang_code:
        return "언어를 선택해 주세요."
    success, msg = model_manager.download_stanza_model(lang_code)
    return msg

def get_installed_status():
    installed = model_manager.get_installed_stanza_models()
    if not installed:
        return "설치된 모델 없음"
    return "\n".join([f"✅ {SUPPORTED_LANGUAGES.get(lang, lang)} ({lang})" for lang in installed])

# Gradio UI 구성
with gr.Blocks(title="PII Shield Global v4 (Granite)") as demo:
    gr.Markdown(f"""# 🛡️ PII Shield Global v4
### Powered by [IBM Granite-4.0-350M]({HF_MODEL_URL}) & Stanza Multi-NLP""")
    
    with gr.Tabs():
        with gr.TabItem("Shield Execution"):
            with gr.Row():
                with gr.Column():
                    input_txt = gr.Textbox(label="Input Text", lines=10, placeholder="마스킹할 텍스트를 입력하세요...")
                    with gr.Row():
                        mask_style = gr.Dropdown(["*", "[TAG]"], label="Masking Style", value="*")
                        minimize_addr = gr.Checkbox(label="Minimize Address (시/구만 표시)", value=False)
                    run_btn = gr.Button("Detect & Mask", variant="primary")
                    dl_auto_status = gr.Textbox(label="Auto-Download Status", interactive=False)
                with gr.Column():
                    output_txt = gr.Textbox(label="Protected Output", lines=10, interactive=False)
                    info_txt = gr.Textbox(label="Language & POS Info", lines=5, interactive=False)
                    entity_label = gr.Label(label="Detected Entities")
            
            run_btn.click(
                process_pii, 
                inputs=[input_txt, mask_style, minimize_addr], 
                outputs=[output_txt, info_txt, entity_label, dl_auto_status]
            )
            
        with gr.TabItem("Settings & Models"):
            gr.Markdown("### 🌐 Stanza Language Model Management")
            with gr.Row():
                lang_options = list(SUPPORTED_LANGUAGES.keys())
                lang_input = gr.Dropdown(
                    choices=lang_options, 
                    label="Select Language to Download",
                    info="Stanza 공식 모델을 다운로드합니다."
                )
                dl_btn = gr.Button("Download Stanza Model", variant="secondary")
            
            dl_status = gr.Textbox(label="Download Status", interactive=False)
            
            gr.Markdown("### 📦 Installed Models")
            refresh_btn = gr.Button("Refresh Installed List")
            installed_txt = gr.Textbox(
                label="Current Models in ~/stanza_resources", 
                lines=5, 
                value=get_installed_status(),
                interactive=False
            )
            
            gr.Markdown(f"""### 🤖 LLM Model Info
- **Model:** `{MODEL_NAME}`
- **Hugging Face:** [Link]({HF_MODEL_URL})
- **Status:** {'🟢 Loaded' if model_manager.model else '🔴 Not Loaded'}

*만약 모델이 로드되지 않았다면, 인터넷 연결을 확인하거나 Hugging Face 로그인이 필요한지 확인하세요.*
""")
            
            dl_btn.click(handle_stanza_download, inputs=[lang_input], outputs=[dl_status])
            refresh_btn.click(get_installed_status, outputs=[installed_txt])

if __name__ == "__main__":
    demo.launch()
