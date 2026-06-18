# PII Shield Global CLI Guide

PII Shield Global 도구를 CLI(Command Line Interface)에서 사용하는 방법입니다.

## 1. 사용법 (Usage)

기본적인 실행 방법은 다음과 같습니다:

```bash
python cli.py "처리할 텍스트 입력"
```

### 파이프라인(Stdin) 사용
긴 텍스트나 파일 내용을 처리할 때 유용합니다:

```bash
cat input.txt | python cli.py
```

## 2. 옵션 (Options)

- `text`: 처리할 텍스트 (생략 시 표준 입력(stdin)으로부터 읽음)
- `--mask-style {*, [TAG]}`: 마스킹 스타일 설정 (기본값: `*`)
- `--minimize-address`: 주소 최소화 모드 활성화 (시/구 정보만 남김)
- `--json`: 결과를 JSON 형식으로 출력
- `--files [FILE_PATTERNS ...]`: 여러 파일을 배치 처리 (와일드카드 `*` 지원)
- `--out-dir DIR`: 배치 처리된 결과를 저장할 디렉토리 (배치 모드 필수)

## 3. 예시 (Examples)

### 기본 마스킹
```bash
python cli.py "제 이름은 홍길동이고 전화번호는 010-1234-5678입니다."
```

### 여러 파일 배치 처리 (Sequential Queue)
특정 폴더의 모든 `.txt` 파일을 순차적으로 처리하여 `output` 폴더에 저장합니다:
```bash
python cli.py --files "./data/*.txt" --out-dir "./output"
```

### TAG 방식 마스킹 및 주소 최소화
```bash
python cli.py "서울시 강남구 테헤란로 123에서 배송 부탁드립니다." --mask-style "[TAG]" --minimize-address
```

### JSON 결과 출력
```bash
python cli.py "My email is test@example.com" --json
```

## 4. 출력 항목
- **Processed Text**: PII가 마스킹된 최종 텍스트
- **Info**: 감지된 언어 정보
- **Entities Found**: 탐지된 PII 항목 리스트 (단어 및 타입)
- **Status**: 처리 상태 메시지
