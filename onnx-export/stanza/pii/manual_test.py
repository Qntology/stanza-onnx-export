import torch
from processor import process_pii
import json

test_text = "안녕하세요, 제 이름은 홍길동이고 전화번호는 010-1234-5678입니다. 이메일은 hong@example.com입니다."
mask_style = "*"

print("--- PII Detection Test ---")
print(f"Input: {test_text}")

# process_pii returns [masked_text, info_text, entity_label, status_msg]
result = process_pii(test_text, mask_style)

print("\n--- Results ---")
print(f"Masked Text: {result[0]}")
print(f"Info: {result[1]}")
print(f"Entities: {result[2]}")
print(f"Status: {result[3]}")
