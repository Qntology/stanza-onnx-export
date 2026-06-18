from processor import process_pii
import json

def test_address_masking():
    test_cases = [
        {
            "text": "서울특별시 종로구 새문안로 89",
            "lang": "ko",
            "desc": "Korean Address (Large to Small)"
        },
        {
            "text": "1600 Amphitheatre Parkway, Mountain View, CA 94043",
            "lang": "en",
            "desc": "English Address (Small to Large)"
        },
        {
            "text": "東京都千代田区丸の内1-9-1",
            "lang": "ja",
            "desc": "Japanese Address (No Spaces)"
        }
    ]

    print("=== Address Masking Verification (minimize_address=True) ===\n")

    for case in test_cases:
        print(f"Description: {case['desc']}")
        print(f"Original: {case['text']}")
        
        # We need to make sure the processor identifies these as ADDRESS.
        # Since process_pii uses Stanza and LLM, we'll just run it.
        masked, lang_info, entities, status = process_pii(case['text'], minimize_address=True)
        
        print(f"Masked:   {masked}")
        print(f"Entities: {entities}")
        print("-" * 50)

if __name__ == "__main__":
    test_address_masking()
