import onnxruntime as ort

def inspect(path):
    print(f"\nModel: {path}")
    try:
        sess = ort.InferenceSession(path)
        print("Inputs:")
        for input in sess.get_inputs():
            print(f"  {input.name}: {input.shape}, {input.type}")
    except Exception as e:
        print(f"Error: {e}")

inspect("onnx_final/en_pos.onnx")
inspect("onnx_final/ko_pos.onnx")
inspect("onnx_final/ja_pos.onnx")
