import onnxruntime as ort

model_path = "onnx_final/en_pos.onnx"
sess = ort.InferenceSession(model_path)

print("Inputs:")
for input in sess.get_inputs():
    print(f"  {input.name}: {input.shape}, {input.type}")

print("\nOutputs:")
for output in sess.get_outputs():
    print(f"  {output.name}: {output.shape}, {output.type}")
