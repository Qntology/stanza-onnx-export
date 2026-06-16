import argparse
import sys
import json
import os
import glob
from processor import process_pii
from model_manager import model_manager

def process_single_text(text, args):
    """단일 텍스트를 처리하고 결과를 반환"""
    masked_text, lang_info, entities, status = process_pii(
        text, 
        mask_char=args.mask_style, 
        minimize_address=args.minimize_address
    )

    if args.json:
        return {
            "masked_text": masked_text,
            "language_info": lang_info,
            "entities": [{"word": str(e[0]), "type": str(e[1])} for e in entities],
            "status": status
        }
    else:
        output = []
        output.append(f"--- Processed Text ---\n{masked_text}\n")
        output.append(f"--- Info ---\n{lang_info}")
        output.append(f"--- Entities Found ---\n{entities}")
        output.append(f"--- Status ---\n{status}")
        return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="PII Shield Global CLI - Detect and Mask PII in text.")
    parser.add_argument("text", nargs="?", help="Text to process. If omitted, reads from stdin.")
    parser.add_argument("--mask-style", choices=["*", "[TAG]"], default="*", help="Masking style (default: *)")
    parser.add_argument("--minimize-address", action="store_true", help="Minimize address (only show City/District)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    # 배치 처리 옵션
    parser.add_argument("--files", nargs="+", help="Process multiple files (glob patterns supported)")
    parser.add_argument("--out-dir", help="Directory to save processed files (batch mode only)")

    args = parser.parse_args()

    # 1. 배치 파일 처리 모드
    if args.files:
        if not args.out_dir:
            print("Error: --out-dir is required when using --files.", file=sys.stderr)
            return

        os.makedirs(args.out_dir, exist_ok=True)
        
        # 파일 목록 수집 (glob 지원)
        all_files = []
        for pattern in args.files:
            all_files.extend(glob.glob(pattern))
        
        if not all_files:
            print("Error: No files found matching the patterns.", file=sys.stderr)
            return

        print(f"📦 Batch Processing: {len(all_files)} files found.")
        
        results = []
        for i, file_path in enumerate(all_files):
            if not os.path.isfile(file_path): continue
            
            print(f"[{i+1}/{len(all_files)}] Processing: {file_path}...")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                res = process_single_text(content, args)
                
                # 저장
                file_name = os.path.basename(file_path)
                out_path = os.path.join(args.out_dir, f"masked_{file_name}")
                
                if args.json:
                    with open(out_path + ".json", "w", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False, indent=2)
                else:
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(res)
                
                results.append({"file": file_path, "status": "Success", "output": out_path})
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")
                results.append({"file": file_path, "status": f"Error: {e}"})

        print(f"✨ Batch Processing Complete. Results saved in '{args.out_dir}'.")
        return

    # 2. 단일 입력 처리 모드
    if args.text:
        input_text = args.text
    else:
        if sys.stdin.isatty():
            parser.print_help()
            return
        input_text = sys.stdin.read()

    if not input_text.strip():
        print("Error: No input text provided.", file=sys.stderr)
        return

    result = process_single_text(input_text, args)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)

if __name__ == "__main__":
    main()
