from processor import process_pii

def test_logistics_pii():
    test_cases = [
        {
            "name": "PCC and Phone Mixed",
            "text": "제 통관번호는 P123456789012이고 휴대폰은 010-5555-6666입니다."
        },
        {
            "name": "Tracking Number and Phone Mixed",
            "text": "CJ대한통운 운송장번호 650123456789 배송 예정입니다. 문의: 02-123-4567"
        },
        {
            "name": "Dense Logistics Info",
            "text": "수령인: 김철수, 주소: 서울시 강남구 테헤란로 123, 송장: 1234567890, 통관부호: P987654321098"
        }
    ]

    print("=== Logistics PII Detection Test ===\n")
    
    for case in test_cases:
        print(f"--- Case: {case['name']} ---")
        print(f"Input: {case['text']}")
        masked_text, _, entities, _ = process_pii(case['text'], minimize_address=True)
        print(f"Masked: {masked_text}")
        print(f"Detected: {entities}\n")

if __name__ == "__main__":
    test_logistics_pii()
