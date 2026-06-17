from processor import process_pii
import json

def test_multilingual_addresses():
    test_cases = [
        {"lang": "ko", "name": "Korean", "text": "서울특별시 서초구 반포대로 23길 45 (삼성동) 우편번호 06655"},
        {"lang": "en", "name": "English (US)", "text": "My office is at 1600 Amphitheatre Parkway, Mountain View, CA 94043, USA."},
        {"lang": "de", "name": "German", "text": "Bitte senden Sie es an Hauptstraße 10, 10115 Berlin, Germany."},
        {"lang": "fr", "name": "French", "text": "L'adresse est 12 Rue de la Paix, 75002 Paris, France."},
        {"lang": "es", "name": "Spanish", "text": "Vivo en Calle de Alcalá 12, 28014 Madrid, España."},
        {"lang": "ja", "name": "Japanese", "text": "住所は東京都港区北青山1-2-3입니다."},
        {"lang": "zh-hans", "name": "Chinese", "text": "中国上海市黄浦区南京路123号"},
        {"lang": "it", "name": "Italian", "text": "Via Roma 15, 00184 Roma RM, Italy."},
        {"lang": "nl", "name": "Dutch", "text": "Morsstraat 111, 2312 BK Leiden, Netherlands."},
        {"lang": "pt", "name": "Portuguese", "text": "Rua da Liberdade 123, 1000-001 Lisboa, Portugal."}
    ]

    print("=== Multilingual Address Detection Test ===\n")
    
    results = []
    for case in test_cases:
        print(f"--- Testing {case['name']} ---")
        print(f"Input: {case['text']}")
        masked_text, _, entities, _ = process_pii(case['text'], minimize_address=True)
        print(f"Masked: {masked_text}")
        print(f"Entities: {entities}\n")
        results.append({
            "name": case['name'],
            "input": case['text'],
            "masked": masked_text,
            "entities": [(str(e[0]), str(e[1])) for e in entities]
        })

    with open("test_multilingual_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    test_multilingual_addresses()
