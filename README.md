# stanza-onnx-export

Stanza(PyTorch) 처리기를 ONNX 로 내보내고, 실행기가 쓸 사전(`vocab.json` 등)을 함께 만듭니다.
대상은 12개 언어입니다: English · German · Spanish · French · Japanese · Portuguese · Arabic · Czech · Italian · Korean · Dutch · Chinese(간체).

```
python final_definitive_conversion.py                        # 12개 언어 → ./onnx_final/<lang>/
python final_definitive_conversion.py --langs en,ko          # 일부만
python verify_onnx.py --langs en,ko,de --json report.json    # PyTorch Stanza 와 대조
python verify_onnx.py --langs ... --dump reference_outputs.json   # 다른 실행기(Rust 등) 대조용 기준 출력
```

모델은 `stanza.Pipeline(..., download_method=REUSE_RESOURCES)` 로 엽니다. 받아 둔 곳은 `STANZA_RESOURCES_DIR` (없으면 `~/stanza_resources`).
받을 때는 `stanza.download(lang, processors="tokenize,mwt,pos,lemma,depparse")` — ja · ko · zh-hans 는 MWT 가 없어 `mwt` 를 빼야 합니다.
2026-11-08 검증은 stanza 1.15.0 · torch 2.8.0 (CPU) · onnxruntime 1.31.1 에서 했습니다.

## 1. 검증 결과 (2026-11-08, 12개 언어)

`verify_onnx.py` 는 세 가지를 잽니다.

1. **원문 끝에서 끝**: `vocab.json` · 딸린 사전과 ONNX 만으로(`RefRuntime` — 실행기가 따를 규약) 원문 → 토큰 · 문장 → MWT 펼침 → 품사 → 원형 → 의존 구문을
   이어 돌리고, 같은 원문을 넣은 `stanza.Pipeline("tokenize[,mwt],pos,lemma,depparse")` 결과와 문장 단위로 비교합니다. 앞 단계 결과를 그대로 다음 단계에 넣습니다.
2. **처리기별**: 파이프라인이 낸 낱말 · upos · xpos · 원형을 넣어 품사 · 원형 · 의존 구문을 따로 잽니다 (앞 단계 오류가 뒤로 번지지 않게).
3. **그래프**: 같은 입력을 넣은 PyTorch 래퍼와 ONNX 출력의 최대 절대 차이 (MWT 는 원본 `decode` · `CharacterClassifier.forward` 와도 비교).

원문은 `verify_sentences.json` 입니다. en · ko 는 일반 문장 10 + 관제 질의 20 + 커머스 · 무역 질의 14, 나머지 10개 언어는 일반 문장 5 + 커머스 · 무역 질의 14
(주문 · 배송 · 쿠폰 · 리뷰 · 선하증권 · 상업송장 · 원산지증명서 · 포장명세서 · 보험증권 · 문서번호). ja · zh 는 공백 없이, 나머지는 보통 글로 씁니다.

| 언어 | 원문 끝에서 끝 (문장 일치) | 낱말 | 처리기별 일치 (%) | 그래프 최대 절대 차이 |
| --- | --- | --- | --- | --- |
| en | 44/44 | 307 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 3.4e-05 · lemma 1.5e-05 · depparse 7.6e-06 · tokenize 7.6e-06 · mwt 9.5e-07 |
| ko | 44/44 | 205 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 6.1e-05 · lemma 1.1e-05 · depparse 1.1e-05 · tokenize 5.7e-06 |
| de | 19/19 | 144 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 1.5e-04 · lemma 1.5e-05 · depparse 4.1e-06 · tokenize 7.6e-06 · mwt 2.9e-06 |
| es | 19/19 | 154 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 6.9e-05 · lemma 3.8e-05 · depparse 6.7e-06 · tokenize 7.6e-06 · mwt 7.6e-06 |
| fr | 19/19 | 157 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 5.3e-05 · lemma 2.3e-05 · depparse 5.7e-06 · tokenize 7.6e-06 · mwt 1.9e-06 |
| ja | 19/19 | 157 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 1.2e-04 · lemma 3.8e-06 · depparse 6.7e-06 · tokenize 7.6e-06 |
| pt | 19/19 | 154 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 1.2e-04 · lemma 1.1e-05 · depparse 5.7e-06 · tokenize 5.7e-06 · mwt 1.9e-06 |
| ar | 19/19 | 122 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 6.1e-05 · lemma 7.6e-06 · depparse 4.8e-06 · tokenize 6.7e-06 · mwt 6.7e-06 |
| cs | 19/19 | 115 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 4.6e-05 · lemma 3.1e-05 · depparse 6.7e-06 · tokenize 7.6e-06 · mwt 1.9e-06 |
| it | 19/19 | 155 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 1.8e-04 · lemma 3.1e-05 · depparse 6.7e-06 · tokenize 7.6e-06 · mwt 3.8e-06 |
| nl | 19/19 | 139 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 1.2e-04 · lemma 1.3e-05 · depparse 5.7e-06 · tokenize 7.6e-06 |
| zh-hans | 19/19 | 115 | upos 100 · xpos 100 · feats 100 · lemma 100 · head 100 · deprel 100 | pos 4.6e-05 · lemma 1.1e-05 · depparse 4.8e-06 · tokenize 7.6e-06 |

원문에서 실제로 펼쳐진 MWT 예: de `vom → von dem` · `zum → zu dem` · `beim → bei dem`, fr `au → à le` · `du → de le` · `des → de les`,
es `del → de el` · `muéstrame → muestra me`, pt `ao → a o` · `do → de o`, it `al → a il` · `dell' → di l'`, ar `برقم → ب رقم` · `لشراء → ل شراء`,
cs `abychom → aby bychom`, en `don't → do n't` · `supplier's → supplier 's`.

### 1.1 이전 판(2026-07 스크립트)과 비교 (en · ko 로 다시 잼)

| | 이전 판 | 이번 판 |
| --- | --- | --- |
| en POS upos | 44.5 % (입력을 가장 바르게 채워도) · 23.2 % (logis-center 입력 방식) | **100 %** |
| ko POS upos | 75.0 % · 45.9 % | **100 %** |
| 원형 (en · ko) | 0 % (93 · 99 % 는 실행 실패) | **100 %** |
| 의존 구문 | en 머리 80.5 % (ROOT 없이) · ko 90 % 문장 실행 실패 | **100 %** |
| 토크나이저 | 특징을 모두 0 으로 넣음 (일본어 · 중국어 사전 특성 없음) | 원문 → 토큰 **100 %** |
| MWT | 없음 | 9개 언어 (en · de · es · fr · pt · ar · cs · it · nl) **100 %** |
| 나머지 10개 언어 | 이전 판(format 1) 사전만 있었음 | 12개 언어 모두 format 2 로 다시 변환 · 대조 |

## 2. 이전 판에서 찾은 결함

| 처리기 | 결함 | 결과 |
| --- | --- | --- |
| POS (ko, 문자 모델) | 문자 LSTM 출력(400칸)을 `trans_char` 투영 없이 앞 125칸만 잘라 넣음 | 품사 입력이 학습 때와 다름 |
| POS (en, 문자 언어모델) | 문장 전체 글자열이 아니라 낱말마다 따로 돌림, 역방향 모델에 정방향 글자, 문자 언어모델 사전 대신 POS 문자 사전 | 영어 품사 44.5 % |
| POS · 의존 구문 | 학습된 LSTM 초기 상태(`*_h_init` · `*_c_init`)를 버림, 문자 모델 주의 합에 채움 칸 포함 | |
| POS | xpos · feats 머리를 내보내지 않음 | 한국어 세종 품사(`ncn+jco`)를 쓸 수 없음 |
| 사전 | 사전학습 단어 사전(25만)이 없어 `pre` 입력이 늘 0, 특수 기호를 첫 글자로 찾으면 `<` 로 겹침 | |
| 원형 | 교사 강제 한 번만 내보냄, 디코더 길이 6 고정, SOS · EOS · 빈도 사전 · 편집 분류 · 복사 없음 | 원형 0 % |
| 의존 구문 | ROOT 자리 없음, xpos 임베딩에 upos 번호, `pos_emb` 두 번 넣기 빠짐, linearization · distance · 대각 마스크 없음 | 한국어 실행 실패 |
| 토크나이저 | 특징(공백 앞 · 대문자 · 숫자 · 문단 처음 · 끝 · 영어 구조 특징 · 일본어 · 중국어 사전 특성) 없음, 끝의 PAD 칸 없음 | |
| MWT | 내보내지 않음 | 프랑스어 `du` · 독일어 `zum` · 아랍어 접어가 한 낱말로 품사 모델에 들어감 |
| 전역 패치 | `torch.max(텐서)` 를 0 으로, `torch.zeros` · `pack_padded_sequence` 를 바꿔치기 | 원형 복사 분기가 한쪽으로 굳음 |
| ONNX 수치 (재검증에서 발견) | `F.logsigmoid` → `Log(Sigmoid(x))` 가 `-inf` → `-inf − (-inf) = NaN` | `고착된` → `고착+되<PAD>+ㄴ` (안정된 식으로) |
| ONNX 연산 | `torch.eye(bool)` → `EyeLike(bool)` 이 onnxruntime 에 없음 | 자리 비교로 |
| 사전 (12개 언어 대조에서 발견) | `WordVocab.ignore` (`['_']`) 를 내보내지 않음. xpos 를 학습하지 않은 fr · pt 는 xpos 가 EMPTY → `'_'` 이고, 의존 구문 입력 `'_'` 는 UNK 가 아니라 EMPTY | fr · pt xpos 0 % · 머리 한 문장씩 어긋남 → `ignore` 를 사전에 적어 100 % |

새 판은 전역 패치 없이 stanza 1.15 의 순전파를 한 문장(batch 1) 기준으로 그대로 옮깁니다 (한 문장이면 PackedSequence 와 채움 없는 텐서가 같음).

## 3. 출력과 입력 규약 (format 2)

`onnx_final/<lang>/` — `vocab.json` 의 `files` 가 처리기별 파일 목록입니다 (실행기 · 앱 내려받기가 이 목록을 따름).

| 파일 | 내용 |
| --- | --- |
| `vocab.json` | 처리기별 사전 · 입출력 이름 · 설정 (`"format": 2`). git 에 넣습니다 |
| `pretrain_vocab.json` | 사전학습 단어 목록 (번호 순, 25만) |
| `lemma_dict.json` | 원형 빈도 사전 `{upos: {낱말: 원형}, "*": {…}}` |
| `mwt_dict.json` | MWT 학습 사전 `{토큰: "낱말 낱말"}` (MWT 가 있는 언어) |
| `tokenize_lexicon.json` | 토크나이저 사전 특성 낱말 목록 (ja · zh-hans, 두 글자 이상) |
| `tokenizer.onnx` · `pos.onnx` · `lemma_encoder.onnx` · `lemma_decoder.onnx` · `depparse.onnx` | 그래프 |
| `mwt_encoder.onnx` · `mwt_decoder.onnx` (seq2seq) 또는 `mwt_classifier.onnx` (en) | MWT 그래프 |

`vocab.json` 밖의 파일은 git 에 넣지 않습니다 (언어마다 0.3~0.8 GB, 12개 언어 5.4 GB).
`reference_outputs.json` 은 `verify_sentences.json` 원문에 대한 PyTorch Stanza 전체 파이프라인 출력이며, 다른 실행기를 같은 기준으로 대조할 때 씁니다
(logis-center `tests/it/stanza_runtime.rs`). 참고 구현: logis-center `src-tauri/src/stanza.rs` (onnxruntime), blueprint-sandbox `core/src/nlp/stanza.rs` (tract).

### 토크나이저 (`tokenizer.onnx`)

1. 문단: `\n[\s\x80-\x9f]*\n` 로 나누고 오른쪽 공백을 지움. 글자마다 공백류(`[\s\x80-\x9f]`)는 `' '`, 연속 공백은 하나만. `skip_newline` (zh-hans)이면 `\n` 을 지움
2. `x` (1, n+1) = 글자 번호(`tokenize.main`, 없으면 UNK) + **끝에 PAD(0) 한 칸** (stanza 는 문단 끝에 PAD 를 붙인 길이로 LSTM 을 돌림)
3. `f` (1, n+1, F) = `tokenize.feat_layout` 순서의 0/1 특징, PAD 칸은 0
   - `space_before` 글자가 공백 · `capitalized` 대문자 · `numeric` `^[\d]+([,\.]+[\d]+)*[,\.]*$` · `end_of_para` · `start_of_para`
   - 구조 특징(en): `labeled_field` · `phone_id` · `date_pattern` · `currency` 정규식이 걸친 글자
   - 사전 특징(ja · zh): 글자 i 에서 앞으로 k 글자(첫 글자 그대로 + 뒤는 소문자)가 사전 낱말이면 `dict_forward_k`, 뒤로 k 글자면 `dict_backward_k`.
     접두(접미) 집합에 없으면 그 방향은 멈춤 (`tokenize_lexicon.json` 에서 낱말 · 접두 · 접미 집합을 만듦)
4. `output` 의 글자별 argmax (0 이어짐 · 1 토큰 끝 · 2 문장 끝 · 3 MWT 끝 · 4 MWT+문장 끝). 마지막 글자는 2 미만이면 2, 2 초과면 4
5. 이메일 · URL 정규식이 걸친 글자는 자르지 않고 마지막 글자만 토큰 끝 (`use_regex_tokens`)
6. 1 이상에서 토큰을 끊음: 왼쪽 공백 제거, 공백류는 `' '`, `replaces_spaces` (ko · ja · zh)면 토큰 안 공백 제거. 2 · 4 에서 문장을 끊음

### MWT (`mwt_*.onnx` + `mwt_dict.json`) — 토크나이저가 3 · 4 로 표시한 토큰만

1. 사전 (`ensemble_dict` · `dict_only`): 그대로 → 토큰이 전부 대문자면 소문자로 찾아 대문자로 → 첫 글자만 대문자면 소문자로 찾아 첫 글자 대문자로
2. `type: seq2seq`: `src_ids` = `[sos] + 글자 + [eos]` (사전 밖 글자는 정렬해 `vocab_size + k`), 인코더 → 디코더 한 걸음씩 (`never_decode_unk` 면 UNK 칸 -inf),
   첫 최댓값, eos 에서 멈춤, 번호 4 미만은 버림. 결과가 비었거나 공백이 없으면 원문 토큰
3. `type: classifier` (en): `logits` (1, L, 2) 에서 글자 자리 1..L-2 마다 `[1] > [0]` 이면 그 글자 앞에서 나눔
4. 펼친 결과를 공백으로 나눠 낱말로. 한 낱말이면 원문 토큰 그대로

### POS (`pos.onnx`)

- `word_ids` (1, n): `pos.word` 번호 (`word_lower` 면 소문자), 없으면 UNK(1)
- `pretrain_ids` (1, n): `pretrain_vocab.json` 에서 **소문자** · 공백을 `\xa0` 로 바꿔 찾음, 없으면 UNK
- 문자 모델 언어 (ko · cs): `char_ids` (n, C) 낱말마다 `pos.char` 번호, 뒤는 0 · `char_lens` (n,)
- 문자 언어모델 언어 (그 밖): `charlm_f_ids` (1, Lf) = `"\n" + 낱말1 + " " + 낱말2 + " " …` 을 `pos.charlm.forward` 로, `charlm_f_offsets` (n,) = 각 낱말 뒤 `" "` 자리.
  역방향 `charlm_b_*` 는 낱말 순서와 글자를 모두 뒤집은 글자열 · 자리를 다시 원래 낱말 순서로 뒤집은 것
- 출력: `upos` (1, n, U), 그 밖의 태그 열은 `tag_order` 순서로. 구성형(`tags.<열>.composite`)은 칸마다 `<열>_<k>` 이고,
  각 칸의 argmax 를 `parts[k]` 목록에서 찾아 EMPTY(2) 칸은 빼고 `sep` 로 잇습니다 (keyed 면 `키=값`, sep 가 빈 문자열이면 글자마다 칸). 아무것도 없으면 `_`.
  목록형 태그에 `pos.ignore.<열>` 이 있으면 EMPTY(2) 번호는 `_` 입니다

### 원형 (`lemma_encoder.onnx` + `lemma_decoder.onnx` + `lemma_dict.json`)

1. `ensemble_dict` 면 `lemma_dict[upos][낱말]` → 없으면 `lemma_dict["*"][낱말]` (caseless 면 소문자)
2. 없으면 탐욕 디코딩: `src_ids` = `[sos] + 글자 + [eos]` (`lemma.char` 번호, 사전에 없는 글자는 정렬해 `vocab_size + k` 로, 최대 `oov_slots` 개),
   `pos_id` = `lemma.pos` 번호. 디코더를 `token = sos` 부터 `max_dec_len` 번까지 한 걸음씩, `log_probs` 의 첫 최댓값을 다음 `token` 으로 (eos 면 멈춤)
3. 편집: `edit_logits` argmax 1 → 낱말 그대로, 2 → 소문자, 0 → 디코딩 결과. 빈 결과나 `<UNK>` 가 든 결과는 낱말 그대로
4. 빈 원형은 `_`

### 의존 구문 (`depparse.onnx`)

0번 자리에 ROOT 를 둡니다: `word_ids` · `lemma_ids` · `upos_ids` · `pretrain_ids` 의 0번은 ROOT(3), `xpos_ids` 는 (1, n+1, P) 구성형이면 ROOT 줄,
`char_ids` 0번 줄은 `[3]` (길이 1), 문자 언어모델은 맨 앞 낱말로 `"\n"` 을 넣습니다. `depparse.ignore` 의 단위(fr · pt 의 xpos `_`)는 UNK 가 아니라 EMPTY(2) 입니다.
출력 `unlabeled` (1, n+1, n+1) 에 행마다 log-softmax, `deprel` 에 관계 축 log-softmax 를 하고
Chu-Liu-Edmonds (루트 칸을 가장 작은 점수 × 길이만큼 내려 단일 루트) → 머리 제약 해소(한 머리에 nsubj · csubj 묶음, obj 가 둘 이상이면 Chu-Liu-Edmonds 를 다시 돌려 고침)로
트리와 관계를 고릅니다 (stanza `decode_graph_predictions(resolve_head_constraints=True)`, `verify_onnx.py` 의 `decode_tree`).

## 4. 하지 않은 것

- 대조 문장은 언어마다 19~44개입니다. 더 넓은 대조는 UD 시험 묶음을 `verify_sentences.json` 형식으로 넣어 돌리면 됩니다.
- 문자 모델이 양방향이거나 주의가 없는 모델, 트랜스포머 임베딩 · 문맥 원형 분류기(영어 `'s` → be/have)를 쓰는 모델은 내보내지 않고 건너뜁니다 (12개 언어 기본 모델에는 없음).
- 토크나이저 문단이 1000 글자를 넘으면 stanza 는 창을 밀며 나눠 돌리는데, 규약은 한 번에 넣는 경우만 다룹니다 (검색 질의용).
- URL 정규식의 `(?:www\.|(?!www))` 앞보기는 Rust `regex` 에 없어 logis-center 실행기는 `(?:www\.)?` 로 둡니다 ('www' 로 시작하지만 'www.' 이 아닌 호스트만 다름).
