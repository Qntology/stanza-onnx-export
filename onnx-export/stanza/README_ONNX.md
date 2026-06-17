# Stanza to ONNX Conversion & Usage Guide

This project provides a definitive tool for converting Stanza NLP models (PyTorch) to ONNX format, optimized for high-performance inference.

## 1. Exporting Models

Use the `stanza_final_onnx_exporter.py` script to convert Stanza's linguistic processors to ONNX.

### Requirements
- `torch >= 2.0`
- `stanza`
- `onnx`, `onnxruntime`
- `requests`

### Usage
```bash
python stanza_final_onnx_exporter.py --lang ko --output_dir ./onnx_final
```

### Supported Processors
- **Tokenizer**: `ko_tokenizer.onnx`
- **POS (Part-of-Speech)**: `ko_pos.onnx` (Includes UPOS, XPOS, and Feats)
- **Depparse (Dependency Parser)**: `ko_depparse.onnx`

## 2. Exported Model Schemas

### POS Model (`ko_pos.onnx`)
- **Inputs**:
  - `word`: Long tensor `[batch, seq]`
  - `wordchars`: Long tensor `[batch*seq, 5]`
  - `pretrained`: Long tensor `[batch, seq]`
  - `word_orig_idx`: Long tensor `[batch*seq]`
  - `sentlens`: Long tensor `[batch]`
  - `wordlens`: Long tensor `[batch*seq]`
- **Outputs**:
  - `upos_logits`: Raw scores for UPOS classes.
  - `xpos_hid`: Hidden representations for XPOS decoding.
  - `ufeats_hid`: Hidden representations for Features decoding.

### Depparse Model (`ko_depparse.onnx`)
- **Inputs**: Similar to POS, but includes `lemma` and `upos/xpos` indices.
- **Outputs**: `unlabeled_scores`, `deprel_scores`.

## 3. Technical Implementation Details

The conversion uses a **Recursive Architectural Patching** strategy:
1. **Linearization**: Overrides Stanza's `PackedLSTM` and `HighwayLSTM` to bypass `PackedSequence` logic, which is incompatible with standard ONNX tracers.
2. **Global Sync**: Patches `CharacterModel` to use pure tensor operations (Sum/Attention) instead of variable-length packing.
3. **Parity**: Maintains exact weights and logical flow while stripping away PyTorch-specific optimizations.

## 4. Troubleshooting

If you encounter `KeyError` or `Dimension Mismatch`:
- Ensure the model package is fully downloaded via `stanza.download('ko')`.
- The exporter assumes the default Stanza model configuration. If using custom models, the patching logic might require adjustment in `stanza_final_onnx_exporter.py`.

---
*Created by Gemini CLI Agent - 2026-06-17*
