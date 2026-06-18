import json
from processor import process_pii

# Load the target text
file_path = "debug_target_text.yaml"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"--- Processing {file_path} ---")

# Process PII
masked_text, lang_info, entities, status = process_pii(text)

# Prepare log content
log_content = []
log_content.append("====================================================")
log_content.append("PII Masking Test Log")
log_content.append(f"Target File: {file_path}")
log_content.append("====================================================")
log_content.append("\n[1] Detected Entities (Names & Organizations/Clubs):")
for word, p_type in entities:
    log_content.append(f"- {word} ({p_type})")

log_content.append("\n[2] Masked Output:")
log_content.append("-" * 50)
log_content.append(masked_text)
log_content.append("-" * 50)

log_content.append(f"\n[3] Status: {status}")
log_content.append(f"Language Info: {lang_info}")

# Save to .txt file
log_file = "debug_masking_log.txt"
with open(log_file, "w", encoding="utf-8") as f:
    f.write("\n".join(log_content))

print(f"--- Log saved to {log_file} ---")
