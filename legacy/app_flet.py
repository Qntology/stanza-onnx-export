import flet as ft
from config import MODEL_NAME, GGUF_MODEL_URL, SUPPORTED_LANGUAGES
from model_manager import model_manager
from processor import process_pii
import threading
import sys
from playwright.sync_api import sync_playwright

# 전역 Playwright 관리 변수
playwright_manager = {
    "sync": None,
    "browser": None,
    "context": None,
    "page": None
}

def main(page: ft.Page):
    page.title = "🛡️ PII Shield Global v4"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.window_width = 1200
    page.window_height = 900

    # 초기화 시 모델 로드
    def initial_load():
        page.splash = ft.ProgressBar()
        page.update()
        print("[System] Loading Models...", file=sys.stderr)
        success, msg = model_manager.load_llm()
        page.splash = None
        if not success:
            page.snack_bar = ft.SnackBar(ft.Text(f"Initial LLM Load Failed: {msg}"))
            page.snack_bar.open = True
        update_llm_status()
        page.update()

    # --- Playwright Handlers (Dedicated Threaded Launch) ---
    def launch_smart_browser(e):
        def _launch():
            if playwright_manager["browser"] is None:
                try:
                    print("[Browser] Attempting to launch Chromium...", file=sys.stderr)
                    playwright_manager["sync"] = sync_playwright().start()
                    playwright_manager["browser"] = playwright_manager["sync"].chromium.launch(headless=False)
                    playwright_manager["context"] = playwright_manager["browser"].new_context()
                    
                    # --- [익스텐션 스타일 스크립트 주입] ---
                    # 페이지가 로드될 때마다 실행될 JS/CSS
                    injection_js = """
                    window.addEventListener('load', () => {
                        const btn = document.createElement('div');
                        btn.innerHTML = '🛡️ PII SCAN';
                        btn.style.position = 'fixed';
                        btn.style.bottom = '20px';
                        btn.style.right = '20px';
                        btn.style.zIndex = '999999';
                        btn.style.backgroundColor = '#2196F3';
                        btn.style.color = 'white';
                        btn.style.padding = '10px 20px';
                        btn.style.borderRadius = '50px';
                        btn.style.cursor = 'pointer';
                        btn.style.boxShadow = '0 4px 15px rgba(0,0,0,0.3)';
                        btn.style.fontWeight = 'bold';
                        btn.style.fontSize = '14px';
                        
                        btn.onclick = () => {
                            btn.style.backgroundColor = '#4CAF50';
                            btn.innerHTML = '⌛ SCANNING...';
                            // Python 쪽에 신호를 보내기 위한 사용자 정의 이벤트
                            window.dispatchEvent(new CustomEvent('pii-scan-trigger'));
                            setTimeout(() => {
                                btn.style.backgroundColor = '#2196F3';
                                btn.innerHTML = '🛡️ PII SCAN';
                            }, 2000);
                        };
                        document.body.appendChild(btn);
                    });
                    """
                    playwright_manager["context"].add_init_script(injection_js)
                    
                    playwright_manager["page"] = playwright_manager["context"].new_page()
                    
                    # Python에서 JS 이벤트를 감지하도록 설정 (익스텐션 통신 역할)
                    def on_scan_triggered():
                        print("[Browser] Scan triggered from injected button!", file=sys.stderr)
                        capture_web_content(None)
                        
                    playwright_manager["page"].expose_binding("trigger_python_scan", lambda source: on_scan_triggered())
                    # JS에서 이벤트를 받았을 때 Python 함수 호출
                    playwright_manager["page"].evaluate("""
                        window.addEventListener('pii-scan-trigger', () => {
                            window.trigger_python_scan();
                        });
                    """)
                    
                    playwright_manager["page"].goto("https://google.com")
                    print("[Browser] Successfully launched!", file=sys.stderr)
                    page.snack_bar = ft.SnackBar(ft.Text("Smart Browser Launched!"))
                    page.snack_bar.open = True
                except Exception as ex:
                    print(f"[Browser] Launch Error: {ex}", file=sys.stderr)
                    page.snack_bar = ft.SnackBar(ft.Text(f"Browser Error: {ex}"))
                    page.snack_bar.open = True
            else:
                print("[Browser] Browser already running.", file=sys.stderr)
                page.snack_bar = ft.SnackBar(ft.Text("Browser is already open."))
                page.snack_bar.open = True
            page.update()
        
        # UI 블로킹 방지를 위해 별도 스레드에서 실행
        threading.Thread(target=_launch, daemon=True).start()

    def capture_web_content(e):
        if playwright_manager["page"]:
            try:
                print("[Browser] Capturing content...", file=sys.stderr)
                # HTML 바디 텍스트 추출 (사용자가 요청한 HTML/텍스트 값 확인)
                content = playwright_manager["page"].evaluate("document.body.innerText")
                input_txt.value = content
                print(f"[Browser] Captured {len(content)} characters.", file=sys.stderr)
                page.snack_bar = ft.SnackBar(ft.Text("Content Captured!"))
                page.snack_bar.open = True
                page.update()
                handle_process(None)
            except Exception as ex:
                print(f"[Browser] Capture Error: {ex}", file=sys.stderr)
                page.snack_bar = ft.SnackBar(ft.Text(f"Capture Error: {ex}"))
                page.snack_bar.open = True
                page.update()
        else:
            page.snack_bar = ft.SnackBar(ft.Text("Please open the browser first!"))
            page.snack_bar.open = True
            page.update()

    # --- Shield Execution Tab Controls ---
    input_txt = ft.TextField(label="Input Text", multiline=True, min_lines=10, hint_text="데이터를 직접 입력하거나 브라우저에서 캡처하세요...", expand=True)
    mask_style = ft.Dropdown(label="Masking Style", value="*", options=[ft.dropdown.Option("*"), ft.dropdown.Option("[TAG]")], width=150)
    minimize_addr = ft.Checkbox(label="Minimize Address", value=False)
    
    def handle_process(e):
        if not input_txt.value: return
        page.splash = ft.ProgressBar()
        page.update()
        masked, info, entities, dl_msg = process_pii(input_txt.value, mask_style.value, minimize_addr.value)
        output_txt.value = masked
        info_txt.value = info
        auto_dl_status.value = dl_msg
        entity_column.controls.clear()
        for word, full_type in entities:
            entity_column.controls.append(ft.ListTile(leading=ft.Icon(ft.Icons.SECURITY), title=ft.Text(word), subtitle=ft.Text(full_type), dense=True))
        page.splash = None
        page.update()

    run_btn = ft.ElevatedButton("Detect & Mask", bgcolor=ft.Colors.PRIMARY, color=ft.Colors.WHITE, on_click=handle_process)
    browser_btn = ft.ElevatedButton("1. Open Smart Browser", icon=ft.Icons.OPEN_IN_NEW, on_click=launch_smart_browser)
    capture_btn = ft.ElevatedButton("2. Capture & Scan Page", icon=ft.Icons.SECURITY, bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE, on_click=capture_web_content)

    output_txt = ft.TextField(label="Protected Output", multiline=True, min_lines=10, read_only=True, expand=True)
    info_txt = ft.TextField(label="Language Info", read_only=True)
    auto_dl_status = ft.TextField(label="Auto-Download Status", read_only=True)
    entity_column = ft.Column(scroll=ft.ScrollMode.AUTO, height=200)

    shield_execution_content = ft.Container(
        padding=10,
        content=ft.Column([
            ft.Row([
                ft.Column([
                    ft.Text("🌐 Web Scanning Tools", weight=ft.FontWeight.BOLD),
                    ft.Row([browser_btn, capture_btn]),
                    ft.Divider(),
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
    lang_input = ft.Dropdown(label="Select Language to Download", options=lang_options, expand=True)
    dl_btn = ft.ElevatedButton("Download Stanza", on_click=lambda e: handle_download())
    dl_status = ft.TextField(label="Download Status", read_only=True)
    refresh_btn = ft.ElevatedButton("Refresh Installed", on_click=lambda e: refresh_installed())
    installed_txt = ft.TextField(label="Current Models", multiline=True, min_lines=5, read_only=True)
    
    llm_status_icon = ft.Icon(ft.Icons.CIRCLE, color=ft.Colors.RED)
    llm_status_text = ft.Text("Not Loaded")

    def update_llm_status():
        is_loaded = model_manager.model is not None
        llm_status_icon.color = ft.Colors.GREEN if is_loaded else ft.Colors.RED
        llm_status_text.value = "🟢 Loaded" if is_loaded else "🔴 Not Loaded"

    def handle_download():
        if not lang_input.value: return
        dl_status.value = f"Downloading {lang_input.value}..."
        page.update()
        success, msg = model_manager.download_stanza_model(lang_input.value)
        dl_status.value = msg
        refresh_installed()
        page.update()

    def refresh_installed():
        installed = model_manager.get_installed_stanza_models()
        installed_txt.value = "\n".join([f"✅ {SUPPORTED_LANGUAGES.get(lang, lang)} ({lang})" for lang in installed]) if installed else "설치된 모델 없음"
        page.update()

    settings_models_content = ft.Container(
        padding=10,
        content=ft.Column([
            ft.Text("🌐 Model Management", size=20, weight=ft.FontWeight.BOLD),
            ft.Row([lang_input, dl_btn]),
            dl_status,
            ft.Divider(),
            ft.Text("📦 Installed Models", size=20, weight=ft.FontWeight.BOLD),
            refresh_btn,
            installed_txt,
            ft.Divider(),
            ft.Text("🤖 LLM Info", size=20, weight=ft.FontWeight.BOLD),
            ft.Row([ft.Column([ft.Text(f"Model: {MODEL_NAME}"), ft.Row([llm_status_icon, llm_status_text])])]),
        ], scroll=ft.ScrollMode.AUTO)
    )

    header = ft.Column([
        ft.Text("🛡️ PII Shield Global v4", size=32, weight=ft.FontWeight.BOLD),
        ft.Markdown(f"Powered by IBM Granite & Playwright Smart Scanner")
    ])

    tabs_control = ft.Tabs(
        length=2,
        content=ft.Column([
            ft.TabBar(tabs=[ft.Tab(label="Shield Execution", icon=ft.Icons.SHIELD), ft.Tab(label="Settings & Models", icon=ft.Icons.SETTINGS)]),
            ft.TabBarView(controls=[shield_execution_content, settings_models_content], expand=True)
        ], expand=True),
        expand=True
    )

    page.add(header, tabs_control)
    refresh_installed()
    initial_load()

if __name__ == "__main__":
    # DeprecationWarning 방지를 위해 최신 권장 방식으로 실행
    ft.app(main)
