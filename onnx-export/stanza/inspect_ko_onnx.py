import onnxruntime as ort

model_path = "onnx_final/ko_pos.onnx"
sess = ort.InferenceSession(model_path)

print("Inputs:")
for input in sess.get_inputs():
    print(f"  {input.name}: {input.shape}, {input.type}")
