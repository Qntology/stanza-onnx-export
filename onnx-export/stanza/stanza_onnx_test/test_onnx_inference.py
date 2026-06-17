import onnxruntime as ort
import numpy as np
import stanza
import os

def test_pos_inference():
    model_path = "../onnx_final/ko_pos.onnx"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"--- Loading ONNX Model: {model_path} ---")
    session = ort.InferenceSession(model_path)
    
    # We need a Stanza pipeline just to use its vocabulary mapping for real text
    # In a production Rust/C++ app, you'd use a JSON mapping instead.
    print("Loading Stanza for vocab mapping...")
    nlp = stanza.Pipeline('ko', processors='tokenize,pos', download_method=None, device='cpu')
    vocab = nlp.processors['pos'].vocab
    
    text = "안녕하세요 반가워요"
    doc = nlp(text)
    
    for sent in doc.sentences:
        words = [w.text for w in sent.words]
        print(f"\nProcessing sentence: {' '.join(words)}")
        
        # 1. Prepare Inputs (Simplied for testing)
        seq_len = len(words)
        word_ids = np.array([[vocab['word'].get(w, vocab['word'].unk_id) for w in words]], dtype=np.int64)
        
        # Dummy chars (in real use, map each char)
        wordchars = np.ones((seq_len, 5), dtype=np.int64) 
        
        pretrained = np.zeros((1, seq_len), dtype=np.int64)
        word_orig_idx = np.arange(seq_len, dtype=np.int64)
        sentlens = np.array([seq_len], dtype=np.int64)
        wordlens = np.ones((seq_len,), dtype=np.int64) * 5
        
        inputs = {
            "word": word_ids,
            "wordchars": wordchars,
            "pretrained": pretrained,
            "word_orig_idx": word_orig_idx,
            "sentlens": sentlens,
            "wordlens": wordlens
        }
        
        # 2. Run Inference
        outputs = session.run(None, inputs)
        upos_logits = outputs[0]
        
        # 3. Map Back to Labels
        preds = np.argmax(upos_logits, axis=-1)[0]
        upos_tags = [vocab['upos'].id2unit(p) for p in preds]
        
        print("Results:")
        for w, tag in zip(words, upos_tags):
            print(f"  {w}: {tag}")

if __name__ == "__main__":
    test_pos_inference()
