import os
import subprocess
import json
import glob

def run_verification():
    print("=== PII Shield: Multilingual Document Verification ===")
    
    test_docs_dir = "test_docs"
    test_results_dir = "test_results"
    os.makedirs(test_results_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(test_docs_dir, "*.txt"))
    
    print(f"Found {len(files)} documents for testing.\n")
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"--- Processing: {file_name} ---")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Run cli.py via subprocess, piping the content
            cmd = ["python", "cli.py", "--json"]
            result = subprocess.run(cmd, input=content, capture_output=True, text=True, encoding="utf-8", errors="replace")
            
            if result.returncode == 0:
                output_json = json.loads(result.stdout)
                
                # Save to test_results
                out_path = os.path.join(test_results_dir, f"masked_{file_name}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output_json, f, ensure_ascii=False, indent=2)
                
                print(f"✅ Success. Language detected: {output_json['language_info']}")
                print(f"   Entities found: {len(output_json['entities'])}")
                for ent in output_json['entities'][:5]: # Show first 5
                    print(f"     - {ent['word']} ({ent['type']})")
                if len(output_json['entities']) > 5:
                    print(f"     ... and {len(output_json['entities'])-5} more.")
            else:
                print(f"❌ Error: {result.stderr}")
        except Exception as e:
            print(f"❌ Exception: {e}")
        print()

    print("=== Verification Complete. Check 'test_results' for detailed JSONs. ===")

if __name__ == "__main__":
    run_verification()
