# Stanza PyTorch to ONNX Conversion Log

## 🛠 Critical Patches Applied

### 1. Traceable Unsort (Korean Fix)
- **Problem**: Stanza uses `sorted(enumerate(oidx))` which is untraceable in ONNX.
- **Fix**: Patched `stanza.models.common.utils.unsort` to use `torch.argsort(oidx)` during tracing.
- **Benefit**: Resolves "too many indices" error for Korean and other CJK models.

### 2. Pad Sequence Patch (English/Multilingual Fix)
- **Problem**: `aten::pad_sequence` is unsupported in ONNX opset 14.
- **Fix**: Global monkey-patch for `torch.nn.utils.rnn.pad_sequence` using a traceable combination of `stack` and `pad`.
- **Benefit**: Allows models with CharLM or BERT-based embeddings to be exported.

### 3. Signature Inspection Wrapper
- **Problem**: Models across different versions/languages have varying argument counts.
- **Fix**: Created `UniversalWrapper` that uses `inspect.signature` to dynamically map inputs to the model's `forward` method.

## 🚀 Language Conversion Status

### [x] Korean (ko)
- **Processors**: Tokenizer, POS, Lemma, Depparse
- **Method**: `kaist_nocharlm` forced.
- **Outcome**: Successfully exported to `./onnx_final/ko_*.onnx`.

### [x] English (en)
- **Processors**: Tokenizer, POS, Lemma, Depparse
- **Method**: Forced `nocharlm` version.
- **Outcome**: Successfully exported to `./onnx_final/en_*.onnx`.

### [ ] Japanese (ja)
- **Status**: Failed (pad_sequence error persists in depparse).
- **Plan**: Investigate if a specific `nocharlm` package exists for depparse or apply direct model code surgery.

## 📈 Accuracy & Parity Progress

| Feature | Python (Stanza) | Rust (ONNX Runtime) | Parity Status |
|---------|-----------------|---------------------|---------------|
| Tokenization | High | High | ✅ Identical |
| POS Tagging | High | High | ✅ Identical |
| Dependency Parsing | High | Integrated | ⚠️ Testing |
| Entity Merging | Advanced (Arc-based) | Ported (Arc-based) | 🔄 In-Progress |
