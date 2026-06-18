# PII Detection & Masking Application (v3)

## 1. 개요 (Overview)
본 프로젝트는 **Qwen3.5-0.8B** 모델과 **Stanza Multi-language** 지원 기능을 결합하여, 사용자의 입력 텍스트 언어를 자동으로 감지하고 해당 언어에 최적화된 PII 탐지를 수행하는 지능형 솔루션입니다.

## 2. 주요 업데이트 (v3)
- **Model Upgrade:** 최신 **Qwen3.5-0.8B** 모델 적용 (Hugging Face 경로 사용).
- **Dynamic Stanza Loading:** 유니코드 및 언어 감지 라이브러리를 통해 메인 언어를 식별하고, 해당 국가의 Stanza 모델을 동적으로 로드합니다.
- **Download Manager:** Gradio UI 내에 별도의 설정(Settings) 탭을 추가하여 사용자가 필요한 언어 모델을 직접 다운로드하고 상태를 확인할 수 있습니다.
- **Unicode Analysis:** 단순 텍스트 감지를 넘어 유니코드 범위를 분석하여 텍스트 내 혼합된 언어 중 지배적인 언어를 식별합니다.

## 3. 기술 스택 (Tech Stack)
- **LLM:** `Qwen/Qwen3.5-0.8B` (또는 지정된 Hugging Face 경로).
- **NLP Engine:** **Stanza** (Stanford NLP).
- **Language Detection:** `langdetect` + Custom Unicode Script Analyzer.
- **UI:** **Gradio** (Multi-tab support).

## 4. 언어 감지 로직
1. **유니코드 분석:** 한글(Hangul), 한자(CJK), 영문(Latin), 키릴(Cyrillic) 등 각 문자 범위별 빈도수를 측정합니다.
2. **최종 결정:** 유니코드 분석 결과와 `langdetect` 결과를 교차 검증하여 Stanza 모델 로드 언어를 결정합니다.

## 5. 모델 관리 시스템
- **자동 다운로드:** LLM 모델이 없을 경우 Hugging Face에서 자동으로 다운로드 시도.
- **수동 다운로드 (UI):** 사용자가 원하는 국가(ko, en, zh, ja, fr 등)를 선택하여 Stanza 모델을 사전에 설치 가능.
