import flet as ft
from config import MODEL_NAME, GGUF_MODEL_URL, SUPPORTED_LANGUAGES
from model_manager import model_manager
from processor import process_pii
import threading

def main(page: ft.Page):
    page.title = "🛡️ PII Shield Global v4 (Flet)"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    # page.scroll = ft.ScrollMode.ADAPTIVE # Removed to avoid conflict with nested scrolls

    # 초기화 시 모델 로드 시도
    def initial_load():
        page.splash = ft.ProgressBar()
        page.update()
        success, msg = model_manager.load_llm()
        page.splash = None
        if not success:
            page.snack_bar = ft.SnackBar(ft.Text(f"Initial LLM Load Failed: {msg}"))
            page.snack_bar.open = True
        update_llm_status()
        page.update()

    # --- Shield Execution Tab Controls ---
    input_txt = ft.TextField(
        label="Input Text", 
        multiline=True, 
        min_lines=10, 
        hint_text="마스킹할 텍스트를 입력하세요...",
        expand=True
    )
    mask_style = ft.Dropdown(
        label="Masking Style", 
        value="*", 
        options=[ft.dropdown.Option("*"), ft.dropdown.Option("[TAG]")],
        width=150
    )
    minimize_addr = ft.Checkbox(label="Minimize Address (시/구만 표시)", value=False)
    run_btn = ft.ElevatedButton("Detect & Mask", bgcolor=ft.Colors.PRIMARY, color=ft.Colors.WHITE, on_click=lambda e: handle_process())
    
    output_txt = ft.TextField(label="Protected Output", multiline=True, min_lines=10, read_only=True, expand=True)
    info_txt = ft.TextField(label="Language & POS Info", read_only=True)
    auto_dl_status = ft.TextField(label="Auto-Download Status", read_only=True)
    entity_column = ft.Column(scroll=ft.ScrollMode.AUTO, height=200)

    def handle_process():
        if not input_txt.value:
            return
        
        page.splash = ft.ProgressBar()
        page.update()
        
        masked, info, entities, dl_msg = process_pii(input_txt.value, mask_style.value, minimize_addr.value)
        
        output_txt.value = masked
        info_txt.value = info
        auto_dl_status.value = dl_msg
        
        entity_column.controls.clear()
        for word, full_type in entities:
            entity_column.controls.append(
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.SECURITY),
                    title=ft.Text(word),
                    subtitle=ft.Text(full_type),
                    dense=True
                )
            )
        
        page.splash = None
        page.update()

    shield_execution_content = ft.Container(
        padding=10,
        content=ft.Column([
            ft.Row([
                ft.Column([
                    input_txt,
                    ft.Row([mask_style, minimize_addr]),
                    run_btn,
                    auto_dl_status
                ], expand=True),
                ft.VerticalDivider(width=1),
                ft.Column([
                    output_txt,
                    info_txt,
                    ft.Text("Detected Entities", weight=ft.FontWeight.BOLD),
                    entity_column
                ], expand=True)
            ], expand=True)
        ], expand=True)
    )

    # --- Settings & Models Tab Controls ---
    lang_options = [ft.dropdown.Option(k, text=f"{v} ({k})") for k, v in SUPPORTED_LANGUAGES.items()]
    lang_input = ft.Dropdown(
        label="Select Language to Download",
        options=lang_options,
        expand=True
    )
    dl_btn = ft.ElevatedButton("Download Stanza Model", on_click=lambda e: handle_download())
    dl_status = ft.TextField(label="Download Status", read_only=True)
    
    refresh_btn = ft.ElevatedButton("Refresh Installed List", on_click=lambda e: refresh_installed())
    installed_txt = ft.TextField(
        label="Current Models in ~/stanza_resources", 
        multiline=True, 
        min_lines=5, 
        read_only=True
    )
    
    llm_status_icon = ft.Icon(ft.Icons.CIRCLE, color=ft.Colors.RED)
    llm_status_text = ft.Text("Not Loaded")

    def update_llm_status():
        is_loaded = model_manager.model is not None
        llm_status_icon.color = ft.Colors.GREEN if is_loaded else ft.Colors.RED
        llm_status_text.value = "🟢 Loaded" if is_loaded else "🔴 Not Loaded"

    def handle_download():
        if not lang_input.value:
            dl_status.value = "언어를 선택해 주세요."
            page.update()
            return
        
        dl_status.value = f"Downloading {lang_input.value}..."
        page.update()
        
        success, msg = model_manager.download_stanza_model(lang_input.value)
        dl_status.value = msg
        refresh_installed()
        page.update()

    def refresh_installed():
        installed = model_manager.get_installed_stanza_models()
        if not installed:
            installed_txt.value = "설치된 모델 없음"
        else:
            installed_txt.value = "\n".join([f"✅ {SUPPORTED_LANGUAGES.get(lang, lang)} ({lang})" for lang in installed])
        page.update()

    settings_models_content = ft.Container(
        padding=10,
        content=ft.Column([
            ft.Text("🌐 Stanza Language Model Management", size=20, weight=ft.FontWeight.BOLD),
            ft.Row([lang_input, dl_btn]),
            dl_status,
            ft.Divider(),
            ft.Text("📦 Installed Models", size=20, weight=ft.FontWeight.BOLD),
            refresh_btn,
            installed_txt,
            ft.Divider(),
            ft.Text("🤖 LLM Model Info", size=20, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.Column([
                    ft.Text(f"Model: {MODEL_NAME}"),
                    ft.Markdown(f"[Hugging Face Link]({GGUF_MODEL_URL})"),
                    ft.Row([llm_status_icon, llm_status_text])
                ])
            ]),
            ft.Text("만약 모델이 로드되지 않았다면, 인터넷 연결을 확인하거나 Hugging Face 로그인이 필요한지 확인하세요.", italic=True, size=12)
        ], scroll=ft.ScrollMode.AUTO)
    )

    # --- Layout Construction ---
    
    header = ft.Column([
        ft.Text("🛡️ PII Shield Global v4", size=32, weight=ft.FontWeight.BOLD),
        ft.Markdown(f"Powered by [IBM Granite-4.0-350M]({GGUF_MODEL_URL}) & Stanza Multi-NLP")
    ])

    tabs_control = ft.Tabs(
        length=2,
        content=ft.Column([
            ft.TabBar(
                tabs=[
                    ft.Tab(label="Shield Execution", icon=ft.Icons.SHIELD),
                    ft.Tab(label="Settings & Models", icon=ft.Icons.SETTINGS),
                ]
            ),
            ft.TabBarView(
                controls=[
                    shield_execution_content,
                    settings_models_content
                ],
                expand=True
            )
        ], expand=True),
        expand=True
    )

    page.add(header, tabs_control)
    
    # Run initial load and refresh
    refresh_installed()
    initial_load()

if __name__ == "__main__":
    ft.app(target=main)
