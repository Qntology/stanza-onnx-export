import os
import json

def update_tracker():
    lang_file = "SUPPORTED_LANGUAGES.md"
    report_file = "batch_conversion_report.json"
    onnx_dir = "onnx_final"
    
    if not os.path.exists(lang_file) or not os.path.exists(report_file):
        return
        
    with open(report_file, "r") as f:
        report = json.load(f)
        
    with open(lang_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    files = os.listdir(onnx_dir)
    new_lines = []
    
    for line in lines:
        if "|" in line and "Code" not in line and "-" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 8:
                lang_code = parts[3]
                if lang_code in report:
                    res = report[lang_code]
                    meta = res["meta"]
                    
                    tok_ok = any(f"{lang_code}_tokenizer.onnx" in f for f in files)
                    pos_ok = any(f"{lang_code}_pos.onnx" in f for f in files)
                    dep_ok = any(f"{lang_code}_depparse.onnx" in f for f in files)
                    
                    parts[4] = "[x]" if tok_ok else "[ ]"
                    parts[5] = "[x]" if pos_ok else "[ ]"
                    parts[6] = "[x]" if dep_ok else "[ ]"
                    
                    if res["success"]:
                        parts[7] = "✅ Success"
                    else:
                        parts[7] = "❌ Failed"
                    
                    # Update Notes with model features
                    notes = f"Pkg: {meta['package']}"
                    if meta['charlm']: notes += ", CharLM"
                    if meta['bert']: notes += ", BERT"
                    parts[8] = notes
                    
                    line = "| " + " | ".join(parts[1:-1]) + " |\n"
        new_lines.append(line)
        
    with open(lang_file, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Updated SUPPORTED_LANGUAGES.md with model metadata.")

if __name__ == "__main__":
    update_tracker()
