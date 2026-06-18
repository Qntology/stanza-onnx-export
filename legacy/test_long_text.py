import os
from model_manager import model_manager
from processor import process_pii

def test_long_complex_text():
    output_file = "test_long_text_result.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Long & Complex Text PII Detection Test ===\n")
        
        # 모델 로드
        success, msg = model_manager.load_llm()
        if not success:
            f.write(f"LLM Load Failed: {msg}\n")
            return

        # 긴 테스트용 텍스트 구성 (한글 + 영어 혼합, 다양한 PII 포함)
        long_text = """
[공지사항] 2026년 신규 프로젝트 참여자 명단 및 안내입니다.

안녕하세요, 프로젝트 매니저 김철수(chulsoo.kim@globaltech.com)입니다. 
이번 'PII Shield' 개발 프로젝트에 참여해주신 모든 분들께 감사드립니다. 

각 파트별 담당자 연락처와 주소 정보를 아래와 같이 공유하오니 확인 부탁드립니다.

1. 개발팀 (Korea Office)
- 팀장: 이영희, 연락처: 010-5555-6666, 주소: 서울특별시 서초구 반포대로 23길 45 (우편번호 06655)
- 담당자: 박지민, 이메일: jimin.park99@naver.com

2. 디자인팀 (Global Support)
- Leader: Sarah Jenkins, Phone: +1-202-555-0147, Email: sarah.j@design-studio.io
- Address: 1600 Amphitheatre Parkway, Mountain View, CA 94043, USA

3. 운영 및 보안팀
- 담당: 최민수, 전화: 02-123-4567, 계좌번호: 신한은행 110-123-456789 (정산용)
- Emergency Contact: Michael O'Connor, Cell: 010-9988-7766

본 메일은 개인정보를 포함하고 있으므로 열람 후 반드시 파기해 주시기 바랍니다.
문의사항은 고객센터(1588-0000) 또는 운영팀으로 연락주세요.

감사합니다.
    """

        f.write(f"\n[Original Text Length]: {len(long_text)} chars\n")
        print(f"Processing text ({len(long_text)} chars)...")
        
        masked_text, pos_info, entities, dl_status = process_pii(long_text, minimize_address=True)
        
        print("Processing complete.")
        
        f.write(f"\n[Address Minimization]: Enabled\n")
        f.write(f"[Detection Status]: {dl_status}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("[Masked Protected Output]\n")
        f.write("="*50 + "\n")
        f.write(masked_text + "\n")
        f.write("="*50 + "\n")
        
        f.write("\n[Detected Entities Summary]\n")
        for word, etype in entities:
            f.write(f"- {word} ({etype})\n")
    
    print(f"Test completed. Results saved to {output_file}")

if __name__ == "__main__":
    test_long_complex_text()
