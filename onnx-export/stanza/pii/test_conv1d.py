import torch
import causal_conv1d
print(f"Causal Conv1d version: {causal_conv1d.__version__}")
x = torch.randn(1, 16, 64).cuda()
weight = torch.randn(16, 4).cuda()
bias = torch.randn(16).cuda()
# Simple test if it doesn't crash
try:
    from causal_conv1d import causal_conv1d_fn
    out = causal_conv1d_fn(x, weight, bias)
    print("causal_conv1d_fn test passed!")
except Exception as e:
    print(f"causal_conv1d_fn test failed: {e}")
