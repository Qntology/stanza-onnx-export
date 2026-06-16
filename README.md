# PII Shield Global

PII Shield Global은 텍스트 내의 개인정보(PII)를 자동으로 감지하고 마스킹하는 도구입니다. **IBM Granite-3.0-350M** 모델과 **Stanza NLP** 엔진을 결합하여 다국어 지원 및 높은 정확도를 제공합니다.

## 주요 기능
- **다국어 감지:** 입력 텍스트의 언어를 자동으로 분석하여 최적의 NLP 모델 적용.
- **정교한 탐지:** 이름, 전화번호, 이메일, 주소, 개인통관고유부호(PCC), 운송장 번호 등 탐지.
- **유연한 마스킹:** `*` 또는 `[TAG]` 스타일의 마스킹 지원.
- **주소 최소화:** 전체 주소 마스킹 또는 시/구 정보만 노출하는 모드 지원.

## 실행 방법

### 1. Web UI (Gradio)
그래픽 인터페이스를 통해 직관적으로 사용하려면 `app.py`를 실행하세요.
```bash
python app.py
```

### 2. CLI (Command Line Interface)
터미널에서 직접 처리하거나 자동화 스크립트에 연동하려면 `cli.py`를 사용하세요.
```bash
python cli.py "처리할 텍스트"
```
자세한 CLI 사용법은 [CLI_GUIDE.md](./CLI_GUIDE.md)를 참조하세요.

## 설치 및 요구 사항
- Python 3.10+
- 필요한 라이브러리: `gradio`, `stanza`, `llama-cpp-python`, `langdetect` 등
- 모델 파일: Hugging Face를 통해 자동으로 다운로드됩니다.

## 파일 구조
- `app.py`: Gradio 기반 웹 인터페이스
- `cli.py`: 터미널 기반 커맨드라인 인터페이스
- `processor.py`: PII 탐지 및 마스킹 핵심 로직
- `model_manager.py`: LLM 및 Stanza 모델 관리
- `config.py`: 설정 및 정규식 정의
- `CLI_GUIDE.md`: CLI 상세 사용 가이드
