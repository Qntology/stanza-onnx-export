import onnxruntime as ort
import numpy as np
import stanza
import os

def test_pos_inference():
    # 경로를 절대 경로로 수정하여 오류 방지
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../onnx_final/ko_pos.onnx"))
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"--- Loading ONNX Model: {model_path} ---")
    session = ort.InferenceSession(model_path)
    
    # 어휘 맵핑을 위해 Stanza 파이프라인 로드 (테스트용)
    print("Loading Stanza for vocab mapping...")
    nlp = stanza.Pipeline('ko', processors='tokenize,pos', download_method=None, device='cpu')
    vocab = nlp.processors['pos'].vocab
    
    text = "안녕하세요 반가워요"
    doc = nlp(text)
    
    for sent in doc.sentences:
        words = [w.text for w in sent.words]
        print(f"\nProcessing sentence: {' '.join(words)}")
        
        # 1. 입력 데이터 준비 (Inference Schema 준수)
        seq_len = len(words)
        # WordVocab은 dict처럼 .get()을 지원하지 않으므로 unit2id를 직접 사용
        word_ids = np.array([[vocab['word'].unit2id.get(w, vocab['word'].unk_id) for w in words]], dtype=np.int64)
        
        # Dummy chars (실제 환경에서는 char vocab 맵핑 필요)
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
        
        # 2. ONNX 추론 실행
        outputs = session.run(None, inputs)
        upos_logits = outputs[0]
        
        # 3. 결과 해석 (ArgMax 및 Label 맵핑)
        preds = np.argmax(upos_logits, axis=-1)[0]
        upos_tags = [vocab['upos'].id2unit(p) for p in preds]
        
        print("Results:")
        for w, tag in zip(words, upos_tags):
            print(f"  {w}: {tag}")

if __name__ == "__main__":
    test_pos_inference()
