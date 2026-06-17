import sys
import json
from processor import process_pii

# 모든 출력을 파일로 리다이렉션하는 클래스
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def run_comprehensive_test():
    report_file = "pii_validation_report.txt"
    sys.stdout = Logger(report_file)

    print("==================================================")
    print("      PII SHIELD COMPREHENSIVE VALIDATION REPORT  ")
    print(f"      Target Model: granite-4.0-h-350m-Q8_0.gguf")
    print("==================================================\n")

    test_scenarios = [
        {
            "category": "1. LONG MIXED CONTEXT (KR/EN)",
            "cases": [
                "안녕하세요, 매니저 김철수입니다. 서울시 서초구 반포대로 23길 45에 거주하며 연락처는 010-1111-2222입니다. Contact Sarah at +1-202-555-0147."
            ]
        },
        {
            "category": "2. MULTILINGUAL ADDRESSES (Sequence-Agnostic)",
            "cases": [
                "US: 1600 Amphitheatre Parkway, Mountain View, CA 94043",
                "DE: Bitte an Hauptstraße 10, 10115 Berlin senden.",
                "FR: 12 Rue de la Paix, 75002 Paris",
                "JP: 東京都港区北青山1-2-3",
                "ES: Calle de Alcalá 12, 28014 Madrid"
            ]
        },
        {
            "category": "3. LOGISTICS SPECIAL NUMBERS (PCC/Tracking)",
            "cases": [
                "통관고유부호: P123456789012, 송장번호: 650123456789",
                "Tracking Number: 987654321098, Phone: 02-123-4567"
            ]
        }
    ]

    for scenario in test_scenarios:
        print(f"### Scenario: {scenario['category']}")
        for text in scenario['cases']:
            print(f"\n[INPUT]: {text}")
            masked, lang, entities, status = process_pii(text, minimize_address=True)
            print(f"[LANG]: {lang}")
            print(f"[MASKED]: {masked}")
            print(f"[ENTITIES]: {entities}")
        print("\n" + "-"*50 + "\n")

    print("==================================================")
    print("      VALIDATION COMPLETE - END OF REPORT        ")
    print("==================================================")

if __name__ == "__main__":
    run_comprehensive_test()
