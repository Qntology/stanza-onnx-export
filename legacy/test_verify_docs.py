import os
import subprocess
import json
import glob

def run_verification():
    print("=== PII Shield: Multilingual Document Verification ===")
    
    test_docs_dir = "test_docs"
    test_results_dir = "test_results"
    log_file_path = "pii_verification_report.txt"
    os.makedirs(test_results_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(test_docs_dir, "*.txt"))
    
    with open(log_file_path, "w", encoding="utf-8") as report:
        report.write("=== PII Shield: Multilingual Document Verification Report ===\n")
        report.write(f"Total documents found: {len(files)}\n\n")

        for file_path in files:
            file_name = os.path.basename(file_path)
            print(f"--- Processing: {file_name} ---")
            report.write(f"--- File: {file_name} ---\n")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 개행 제거하여 연속된 텍스트로 만듦
                clean_content = content.replace("\n", " ").replace("\r", "")
                clean_content = " ".join(clean_content.split())
                
                report.write(f"[Original Text]\n{clean_content}\n\n")
                    
                # Run cli.py via subprocess, piping the content
                cmd = ["python", "cli.py", "--json"]
                result = subprocess.run(cmd, input=clean_content, capture_output=True, text=True, encoding="utf-8", errors="replace")
                
                if result.returncode == 0:
                    output_json = json.loads(result.stdout)
                    
                    report.write(f"[Masked Text]\n{output_json['masked_text']}\n\n")
                    report.write("[Entities Detected]\n")
                    for ent in output_json['entities']:
                        report.write(f"- {ent['word']} ({ent['type']})\n")
                    report.write("\n")
                    
                    # Save to test_results
                    out_path = os.path.join(test_results_dir, f"masked_{file_name}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(output_json, f, ensure_ascii=False, indent=2)
                    
                    print(f"✅ Success. Entities found: {len(output_json['entities'])}")
                else:
                    report.write(f"❌ Error during processing: {result.stderr}\n\n")
                    print(f"❌ Error: {result.stderr}")
            except Exception as e:
                report.write(f"❌ Exception: {e}\n\n")
                print(f"❌ Exception: {e}")
            report.write("-" * 80 + "\n\n")

    print(f"\n=== Verification Complete. Report saved to: {log_file_path} ===")

    print("=== Verification Complete. Check 'test_results' for detailed JSONs. ===")

if __name__ == "__main__":
    run_verification()
