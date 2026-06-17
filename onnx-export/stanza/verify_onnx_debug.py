import onnxruntime as ort
import numpy as np
import os

def verify_depparse():
    model_path = "onnx_final/ko_depparse.onnx"
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    print(f"Verifying {model_path}...")
    try:
        session = ort.InferenceSession(model_path)
        print("✅ Model loaded successfully.")
        
        # 입력 이름 확인
        for input in session.get_inputs():
            print(f"Input: {input.name}, Shape: {input.shape}, Type: {input.type}")
            
        # 가짜 입력 생성 (batch=1, seq=5)
        batch, seq = 1, 5
        inputs = {
            "word": np.ones((batch, seq), dtype=np.int64),
            "wordchars": np.ones((seq, 5), dtype=np.int64),
            "upos": np.ones((batch, seq), dtype=np.int64),
            "xpos": np.ones((batch, seq, 1), dtype=np.int64), # kaist는 xpos가 composite일 수 있음
            "ufeats": np.ones((batch, seq, 1), dtype=np.int64),
            "pretrained": np.ones((batch, seq), dtype=np.int64),
            "lemma": np.ones((batch, seq), dtype=np.int64),
            "head": np.ones((batch, seq - 1), dtype=np.int64),
            "word_orig_idx": np.arange(seq, dtype=np.int64),
            "sentlens": np.array([seq], dtype=np.int64),
            "wordlens": np.ones((seq,), dtype=np.int64) * 5
        }
        
        # 모델마다 xpos_num, feats_num이 다를 수 있으므로 세션 정보를 바탕으로 재조정
        input_names = [i.name for i in session.get_inputs()]
        filtered_inputs = {k: v for k, v in inputs.items() if k in input_names}
        
        # xpos/ufeats 차원 보정
        for inp in session.get_inputs():
            if inp.name in ["xpos", "ufeats"] and len(inp.shape) == 3:
                dim = inp.shape[2]
                if isinstance(dim, str): dim = 1 # dynamic
                filtered_inputs[inp.name] = np.ones((batch, seq, dim), dtype=np.int64)

        outputs = session.run(None, filtered_inputs)
        print("✅ Inference successful!")
        print(f"Output shapes: {[o.shape for o in outputs]}")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    verify_depparse()
