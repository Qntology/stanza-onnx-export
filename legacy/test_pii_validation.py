
from model_manager import model_manager
from processor import process_pii

def test_pii_validation():
    test_cases = [
        {
            "name": "Korean Address & Phone",
            "text": "제 주소는 서울특별시 강남구 테헤란로 123길 45 (삼성동)이고, 전화번호는 010-1234-5678입니다."
        },
        {
            "name": "English Address & Phone",
            "text": "My office is located at 1234 Market St, San Francisco, CA 94103, USA. You can call me at +1-415-555-0199."
        },
        {
            "name": "Mixed Address & Phone",
            "text": "배송지는 경기도 성남시 분당구 판교역로 166 (우편번호 13487)입니다. 문의: 02-555-1212. International contact: 44-20-7946-0958."
        }
    ]

    print("=== PII Masking Validation Test ===\n")
    
    # Ensure model is loaded
    success, msg = model_manager.load_llm()
    if not success:
        print(f"LLM Load Failed: {msg}")
        return

    for case in test_cases:
        print(f"--- Case: {case['name']} ---")
        print(f"Original: {case['text']}")
        
        masked_text, _, entities, _ = process_pii(case['text'], minimize_address=True)
        
        print(f"Masked:   {masked_text}")
        print("Detected Entities:")
        for word, etype in entities:
            print(f"  - {word} ({etype})")
        print("\n")

if __name__ == "__main__":
    test_pii_validation()
