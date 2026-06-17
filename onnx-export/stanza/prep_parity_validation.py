import stanza
import os
import torch
import torch.nn as nn
import json

# --- 1. THE DUMMY LAYER INJECTION ---

class DummyCharModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
    def forward(self, *args, **kwargs):
        # Find batch/seq from any tensor arg
        for a in args:
            if torch.is_tensor(a) and a.dim() >= 1:
                return torch.zeros((a.size(0), self.output_dim), device=a.device)
        return torch.zeros((1, self.output_dim))

def apply_dummy_patch(model):
    if hasattr(model, 'charmodel') and model.charmodel is not None:
        # Check output dimension from the transformation layer
        out_dim = 100 # default
        if hasattr(model, 'trans_char'):
            out_dim = model.trans_char.weight.shape[1]
        model.charmodel = DummyCharModel(out_dim)
        print(f"      Patched with DummyCharModel (dim={out_dim})")

# ... (UniversalWrapper remains the same) ...

# --- 2. ACCURACY VALIDATOR PREP (Rust to Python Parity) ---

def generate_rust_test_script():
    """Generates a helper to run Rust and compare results with pii/test_results."""
    with open("verify_parity.py", "w", encoding="utf-8") as f:
        f.write('''
import os
import difflib

def compare_results():
    py_dir = "pii/test_results"
    rs_dir = "stanza_onnx_test/test_results_hybrid"
    
    if not os.path.exists(rs_dir):
        print("❌ Rust results directory not found.")
        return

    print(f"\\n{'Language':<10} | {'File':<30} | {'Parity Status':<15}")
    print("-" * 65)
    
    for f_name in os.listdir(py_dir):
        if not f_name.endswith(".txt"): continue
        
        py_path = os.path.join(py_dir, f_name)
        # Rust files follow hybrid_masked_{original_name}
        rs_name = f"hybrid_masked_{f_name.replace('masked_', '')}"
        rs_path = os.path.join(rs_dir, rs_name)
        
        if os.path.exists(rs_path):
            with open(py_path, 'r', encoding='utf-8') as f1, open(rs_path, 'r', encoding='utf-8') as f2:
                py_text = f1.read().strip()
                rs_text = f2.read().strip()
                
                if py_text == rs_text:
                    status = "✅ 100% Match"
                else:
                    ratio = difflib.SequenceMatcher(None, py_text, rs_text).ratio()
                    status = f"⚠️ {ratio:.1%} Match"
            
            lang = f_name.split('_')[1] if '_' in f_name else '??'
            print(f"{lang:<10} | {f_name:<30} | {status:<15}")

if __name__ == "__main__":
    compare_results()
''')

# (Main process_lang will be updated in next turn to use Dummy and Acc-check)
