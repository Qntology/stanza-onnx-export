import stanza
import time
import requests

langs = ['ja', 'zh-hans', 'vi', 'en', 'ar', 'es']

def download_with_retry(lang, retries=3):
    for i in range(retries):
        try:
            print(f"[{i+1}/{retries}] Attempting to download Stanza model for: {lang}...")
            stanza.download(lang, processors='tokenize,pos,lemma,depparse')
            print(f"✅ Successfully downloaded: {lang}")
            return True
        except Exception as e:
            print(f"⚠️ Error downloading {lang}: {e}")
            if i < retries - 1:
                print("Waiting 5 seconds before retry...")
                time.sleep(5)
    return False

for lang in langs:
    success = download_with_retry(lang)
    if not success:
        print(f"❌ Failed to download {lang} after multiple attempts. Skipping...")

print("\n--- Download Phase Finished ---")
