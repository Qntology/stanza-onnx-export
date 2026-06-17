import os
import json
import difflib

def compare_results():
    py_dir = "pii/test_results"
    rs_dir = "stanza_onnx_test/test_results_hybrid"
    
    if not os.path.exists(rs_dir):
        print("❌ Rust results directory not found.")
        return

    print(f"\n{'Language':<10} | {'File':<30} | {'Parity Status':<15}")
    print("-" * 65)
    
    match_count = 0
    total_count = 0
    
    for py_file in os.listdir(py_dir):
        if not py_file.endswith(".json"): continue
        
        # Original filename from masked_en_report.txt.json -> en_report.txt
        raw_name = py_file.replace("masked_", "").replace(".json", "")
        rs_file = f"hybrid_masked_{raw_name}"
        rs_path = os.path.join(rs_dir, rs_file)
        
        if os.path.exists(rs_path):
            total_count += 1
            with open(os.path.join(py_dir, py_file), 'r', encoding='utf-8') as f1, \
                 open(rs_path, 'r', encoding='utf-8') as f2:
                
                try:
                    py_data = json.load(f1)
                    py_text = py_data.get("masked_text", "").strip()
                    rs_text = f2.read().strip()
                    
                    if py_text == rs_text:
                        status = "✅ 100% Match"
                        match_count += 1
                    else:
                        ratio = difflib.SequenceMatcher(None, py_text, rs_text).ratio()
                        status = f"⚠️ {ratio:.1%} Match"
                except:
                    status = "❌ JSON Error"
            
            lang = py_file.split('_')[1] if '_' in py_file else '??'
            print(f"{lang:<10} | {py_file:<30} | {status:<15}")

    if total_count > 0:
        print(f"\n🚀 Overall Parity Score: {match_count/total_count:.1%} ({match_count}/{total_count} files identical)")
        print("Note: Partial matches are often due to minor tokenization differences in whitespace.")

if __name__ == "__main__":
    compare_results()
