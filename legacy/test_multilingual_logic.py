from processor import process_pii
import json

def test_multilingual_pii():
    test_cases = [
        {
            "lang": "English",
            "text": "My name is Jose Mourinho and I live at 123 Baker Street in London. You can email me at jose@example.com."
        },
        {
            "lang": "Spanish",
            "text": "Hola, soy Sergio Ramos. Mi dirección es Calle de Alcalá, 1, 28014 Madrid, España."
        },
        {
            "lang": "Japanese",
            "text": "私の名前は田中太郎です。東京都渋谷区神南1-2-3に住んでいます。"
        },
        {
            "lang": "French",
            "text": "Je m'appelle Jean Dupont. J'habite au 10 Rue de la Paix, 75002 Paris."
        }
    ]
    
    results = []
    print("--- Starting Multilingual PII Test ---")
    
    for case in test_cases:
        print(f"\n[Processing {case['lang']}...]")
        masked_text, lang_info, entities, status = process_pii(case["text"])
        
        print(f"Detected {lang_info}")
        print(f"Entities: {entities}")
        print(f"Masked: {masked_text}")
        
        results.append({
            "language": case["lang"],
            "detected_lang": lang_info,
            "original": case["text"],
            "masked": masked_text,
            "entities": [{"word": str(e[0]), "type": str(e[1])} for e in entities]
        })

    with open("test_multilingual_logic_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n--- Multilingual Test Complete. Results saved to test_multilingual_logic_result.json ---")

if __name__ == "__main__":
    test_multilingual_pii()
