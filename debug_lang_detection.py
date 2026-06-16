from utils import analyze_unicode_script, get_best_language
import os

test_docs_dir = "test_docs"
for f in os.listdir(test_docs_dir):
    if f.endswith(".txt"):
        with open(os.path.join(test_docs_dir, f), "r", encoding="utf-8") as file:
            content = file.read()
            analysis = analyze_unicode_script(content)
            lang, subs = get_best_language(content)
            print(f"File: {f}")
            print(f"  Scores: {analyze_unicode_script(content)}")
            print(f"  Detected: {lang} | Subs: {subs}")
