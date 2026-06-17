# Stanza PyTorch to ONNX Conversion Guide

This document describes the process and technical requirements for converting Stanza NLP models to ONNX for use in the Rust-based PII detection pipeline.

## 1. Overview
Stanza models use PyTorch (`.pt`). To achieve high-speed inference in Rust without a Python dependency, we export these models to ONNX using `torch.onnx`.

## 2. Key Challenges & Solutions

### A. Dynamic Input Signatures
Different languages have different model architectures:
- **Tokenizer**: Some require only character IDs, others require character IDs + feature vectors. The RNN input dimension varies by language.
- **POS (Part-of-Speech)**: Some models use only word IDs and pretrained embeddings. Others (like Korean) require `wordchars` (character-level features).

**Solution**: The exporter scripts (`export_tokenizers.py`, `export_pos_universal.py`) dynamically inspect the PyTorch model's weight shapes to determine the required input dimensions and feature flags before exporting.

### B. Embedding Gaps
Standard Stanza models rely on massive external word vectors (Glove, FastText). Converting these directly to ONNX makes models too large (>500MB).
**Solution**: We extract the `word2id` and `pre2id` (pretrained mapping) from the Stanza checkpoints and store them in small JSON vocabs. The Rust implementation uses these to feed the correct IDs to the ONNX model's internal embedding layer.

### C. Input Ordering
The `forward()` method signature of Stanza models is complex.
**Solution**: We use `Wrapper` classes (inheriting from `nn.Module`) to expose a simplified, stable interface for ONNX Runtime.

## 3. Conversion Workflow
1. **Download**: `stanza.download(lang)`
2. **Vocab Extraction**: `extract_vocabs_v3.py` (Captures character, word, and pretrained mappings).
3. **Tokenizer Export**: `export_tokenizers_v2.py` (Detects feature dims and exports `[lang]_tokenizer.onnx`).
4. **POS Export**: `export_pos_universal.py` (Handles optional `wordchars` and exports `[lang]_pos.onnx`).

## 4. Verification
After conversion, models should be tested using the Rust `cargo run` command to ensure:
- Tagging accuracy matches the Python version.
- Multi-byte characters (CJK) are correctly handled.
- Memory usage is within reasonable limits.
