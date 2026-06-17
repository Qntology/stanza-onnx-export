import sys
import os
from processor import process_pii

def run_test():
    logistics_data = """
[배송지시서 - CJ대한통운]
주문번호: 20260616-0001928
운송장번호: 5678-1234-9012
수령인: 김철수 (Kim Chul-soo)
연락처: 010-9876-5432 / 02-555-1234
주소: 서울특별시 송파구 가락동 123-45 한라아파트 101동 502호 (우편번호: 05800)
배송요청사항: 문 앞에 놓아주세요. 비밀번호 1234*

[인보이스 - 해외직구]
Sender: Global Logistics Inc. (John Smith)
Address: 123 Logistics Way, Long Beach, CA 90802, USA
Recipient: 이영희 (Lee Young-hee)
ID Card Number: 950101-2034567
Phone: +82-10-1111-2222
Email: younghee.lee88@gmail.com
Items: 
1. Mechanical Keyboard (KBD-99) - $120.00
2. Gaming Mouse (G-PRO) - $80.00
Total: $200.00

[창고 입고 내역]
입고담당자: 박지민 대리
연락처: jimin.park@logis-pro.co.kr
창고위치: 경기도 이천시 마장면 덕평로 123번길 45 (이천 제2센터)
계좌번호: 국민은행 123-456-78-901234 (예금주: 주식회사 로지스프로)
"""

    commerce_data = """
[주문 상세 내역 - 네이버쇼핑]
구매일시: 2026-06-16 10:30:45
구매자: 최민준
아이디: mjchoi_top1
휴대폰: 010-4444-5555
환불계좌정보: 신한은행 110-222-333333 (예금주: 최민준)

[고객 상담 로그 - 반품 문의]
상담원: 안녕하세요, 무엇을 도와드릴까요?
고객(한소희): 네, 제가 지난주에 주문한 원피스 사이즈가 안 맞아서 반품하고 싶어요.
상담원: 네, 성함과 주문번호 확인 부탁드립니다.
고객(한소희): 한소희고요, 주문번호는 99887766입니다. 주소는 부산광역시 해운대구 우동 555-1번지 아이파크 2동 3001호예요.
고객(한소희): 제 이메일로 반품 택배 접수증 보내주세요. hsh_dress@naver.com 입니다.
상담원: 확인 감사합니다. 등록된 번호 010-8888-9999로 안내 문자 발송해드렸습니다.

[결제 영수증]
상호: (주)커머스월드
대표자: 정다은
사업장 소재지: 서울시 강남구 테헤란로 99길 10, 5층
전화번호: 02-1234-5678
결제수단: 현대카드 (****-****-1234-5678)
결제금액: 59,000원
주민등록번호(현금영수증): 881212-1******
"""

    print("=== [LOGISTICS DATA TEST] ===")
    masked_logistics, pos_logistics, pii_logistics, msg_logistics = process_pii(logistics_data)
    print(f"Status: {msg_logistics}")
    print("\n[Detected PII]")
    for word, p_type in pii_logistics:
        print(f"- {word} ({p_type})")
    print("\n[Masked Text]")
    print(masked_logistics)

    print("\n\n" + "="*30 + "\n\n")

    print("=== [COMMERCE DATA TEST] ===")
    masked_commerce, pos_commerce, pii_commerce, msg_commerce = process_pii(commerce_data)
    print(f"Status: {msg_commerce}")
    print("\n[Detected PII]")
    for word, p_type in pii_commerce:
        print(f"- {word} ({p_type})")
    print("\n[Masked Text]")
    print(masked_commerce)

if __name__ == "__main__":
    run_test()
