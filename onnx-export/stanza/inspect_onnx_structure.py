import onnx

def inspect_onnx_tensors(path):
    print(f"--- Inspecting Tensors in {path} ---")
    model = onnx.load(path)
    
    print("\n[Initializers (Weights)]")
    initializers = model.graph.initializer
    # Sort by size to see the biggest ones
    sorted_init = sorted(initializers, key=lambda x: np.prod(x.dims) if hasattr(x, 'dims') else 0, reverse=True)
    
    for i in initializers[:20]: # Show first 20
        dims = list(i.dims)
        size = 1
        for d in dims: size *= d
        print(f"  Name: {i.name}, Dims: {dims}, Size: {size} elements (~{size*4/1024/1024:.2f} MB if float32)")

    print("\n[Inputs]")
    for input in model.graph.input:
        print(f"  Name: {input.name}")

if __name__ == "__main__":
    import numpy as np
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "onnx_final/ko_pos.onnx"
    inspect_onnx_tensors(path)
