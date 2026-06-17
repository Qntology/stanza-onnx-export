# Stanza to ONNX Technical Conversion Guide

이 문서는 Stanza PyTorch 모델(`.pt`)을 Rust PII 엔진에서 사용 가능한 ONNX 모델로 변환하는 기술적 원리와 절차를 설명합니다.

## 1. 핵심 연관 파일 및 참조 소스

변환 로직은 `stanza-main` 라이브러리의 다음 핵심 소스 코드를 분석하여 구현되었습니다.

| 모델 타입 | Stanza 참조 소스 (stanza-main/) | 주요 참조 로직 |
|-----------|-------------------------------|----------------|
| **Tokenizer** | `stanza/models/tokenize/model.py` | `BaseTrainer`, `Model` 클래스의 RNN 입력 구조 |
| **POS** | `stanza/models/pos/model.py` | `Tagger.forward()`의 인자 순서 및 CharLM 연동 |
| **Depparse** | `stanza/models/depparse/model.py` | `GraphParser.forward()`의 Head/Deprel 예측 로직 |
| **공통 유틸** | `stanza/models/common/utils.py` | `unsort` 함수 (텐서 정렬 복원 로직) |
| **Scorer** | `stanza/models/common/biaffine.py` | `BiaffineScorer` 레이어의 텐서 연산 |

## 2. 주요 해결 과제 및 패치 원리

### A. Traceable Unsort (한국어 등 CJK 필수)
- **참조**: `stanza/models/common/utils.py`의 `unsort` 함수.
- **문제**: 원본 코드는 파이썬 `sorted()` 함수를 사용하여 텐서를 정렬하는데, 이는 ONNX의 정적 그래프에 기록되지 않습니다.
- **해결**: `torch.argsort()`를 사용하는 순수 텐서 연산으로 교체하여 Tracing이 가능하도록 패치했습니다.

### B. `pad_sequence` 우회 (다국어 공통)
- **문제**: `aten::pad_sequence` 연산은 ONNX에서 지원되지 않습니다.
- **해결**: `torch.nn.functional.pad`와 `torch.stack`을 조합한 커스텀 함수로 `torch.nn.utils.rnn.pad_sequence`를 몽키 패치하여 ONNX 가 호환 가능한 연산으로 변환합니다.

### C. 동적 시그니처 대응 (`UniversalWrapper`)
- **문제**: 언어마다 `forward()` 메서드가 받는 인자 개수(text, lemma 포함 여부 등)가 다릅니다.
- **해결**: 파이썬의 `inspect.signature`를 사용하여 런타임에 모델의 인자를 분석하고, 필요한 값만 동적으로 채워주는 래퍼 클래스를 구현했습니다.

## 3. 실행 방법

### 일괄 변환 실행
```powershell
python batch_convert_multilingual.py
```
- 이 스크립트는 52개 언어를 순회하며 모델 다운로드 -> 패치 적용 -> ONNX 익스포트 -> Vocab 추출을 자동으로 수행합니다.

### 현황 업데이트
```powershell
python auto_update_tracker.py
```
- `onnx_final` 폴더의 결과물을 바탕으로 `SUPPORTED_LANGUAGES.md` 체크박스를 갱신합니다.

## 4. 관련 파일 구조
- `batch_convert_multilingual.py`: 메인 변환 로직.
- `auto_update_tracker.py`: MD 파일 현황 업데이트.
- `onnx_final/`: 생성된 `.onnx` 파일 저장소.
- `vocabs/`: Rust에서 사용할 인덱스-태그 매핑 JSON 저장소.
