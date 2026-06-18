# Multilingual Stanza ONNX Conversion Report

이 문서는 Stanza 모델의 다국어 ONNX 변환 및 Rust 엔진 통합 현황을 언어별로 상세히 기록합니다.

## 1. 모델 유형 속성 가이드
- **[A] NoCharLM**: CharLM 레이어가 없어 ONNX 변환이 가장 안정적입니다.
- **[B] CharLM/BERT**: CharLM 또는 BERT가 포함되어 있어 변환 시 `pad_sequence` 패치가 필요합니다.
- **[C] Skeleton(Defensive)**: 변환 에러 발생 시 CharLM 레이어를 `Identity` 더미 모듈로 교체하여 생성된 모델입니다.

## 2. 언어별 상세 현황

| 언어 (Code) | 패키지 유형 | 변환 상태 | 대응 전략 |
| :--- | :--- | :---: | :--- |
| **sq** (Albanian) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **ar** (Arabic) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **az** (Azerbaijani) | - | ❌ 실패 | 지원 프로세서 부재 |
| **bn** (Bengali) | - | ❌ 실패 | 지원 프로세서 부재 |
| **bg** (Bulgarian) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **ca** (Catalan) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **zh-hans** (Chinese) | [A] | ✅ 성공 | 표준 변환 |
| **hr** (Croatian) | [A] | ✅ 성공 | 표준 변환 |
| **cs** (Czech) | [A] | ✅ 성공 | 표준 변환 |
| **da** (Danish) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **nl** (Dutch) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **en** (English) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **et** (Estonian) | [A] | ✅ 성공 | 표준 변환 |
| **fi** (Finnish) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **fr** (French) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **ka** (Georgian) | [A] | ✅ 성공 | 표준 변환 |
| **de** (German) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **el** (Greek) | [A] | ✅ 성공 | 표준 변환 |
| **he** (Hebrew) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **hi** (Hindi) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **hu** (Hungarian) | [A] | ✅ 성공 | 표준 변환 |
| **is** (Icelandic) | [A] | ✅ 성공 | 표준 변환 |
| **id** (Indonesian) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **it** (Italian) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **ja** (Japanese) | [B] | ✅ 성공 | Skeleton 모드 (Dummy Layer) |
| **kk** (Kazakh) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **km** (Khmer) | - | ❌ 실패 | 지원 프로세서 부재 |
| **ko** (Korean) | [A] | ✅ 성공 | `unsort` 패치, 고정 패키지 |
| **lv** (Latvian) | [A] | ✅ 성공 | 표준 변환 |
| **lt** (Lithuanian) | [A] | ✅ 성공 | 표준 변환 |
| **ms** (Malay) | - | ❌ 실패 | 지원 프로세서 부재 |
| **mr** (Marathi) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **no** (Norwegian) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **fa** (Persian) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **pl** (Polish) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **pt** (Portuguese) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **ro** (Romanian) | [A] | ✅ 성공 | 표준 변환 |
| **ru** (Russian) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **sr** (Serbian) | [A] | ✅ 성공 | 표준 변환 |
| **sk** (Slovak) | [A] | ✅ 성공 | 표준 변환 |
| **sl** (Slovenian) | [A] | ✅ 성공 | 표준 변환 |
| **es** (Spanish) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **sw** (Swahili) | - | ❌ 실패 | 지원 프로세서 부재 |
| **sv** (Swedish) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **tl** (Tagalog) | - | ❌ 실패 | 지원 프로세서 부재 |
| **te** (Telugu) | [A] | ✅ 성공 | 표준 변환 |
| **th** (Thai) | [A] | ✅ 성공 | 표준 변환 |
| **tr** (Turkish) | [A] | ✅ 성공 | 표준 변환 |
| **uk** (Ukrainian) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **ur** (Urdu) | [B] | ✅ 성공 | `pad_sequence` 패치 적용 |
| **uz** (Uzbek) | [A] | ✅ 성공 | 표준 변환 |
| **vi** (Vietnamese) | [B] | ⚠️ 에러 | 프로세서 정의 누락 |

## 3. Rust 엔진 활용 가이드
- 모든 성공된 언어의 ONNX 파일은 `./onnx_final/` 디렉토리에 존재합니다.
- Rust 엔진(`stanza_onnx_test`)은 `HybridPiiProcessor` 내의 언어 맵핑을 통해 자동으로 올바른 모델을 로드합니다.
- 모델 로드 시 파일이 없으면 자동으로 해당 언어를 건너뛰도록 설계되었습니다.
