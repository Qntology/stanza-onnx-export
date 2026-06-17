import stanza

langs = ['ja', 'zh-hans', 'vi', 'en', 'ar', 'es']
for lang in langs:
    print(f"\n--- Downloading Stanza model for: {lang} ---")
    try:
        if lang == 'vi':
            # Vietnamese might not have all processors
            stanza.download(lang, processors='tokenize,pos,depparse')
        else:
            stanza.download(lang, processors='tokenize,pos,lemma,depparse')
    except Exception as e:
        print(f"Error downloading {lang}: {e}")
        # Fallback to default
        stanza.download(lang)

print("\nAll models download attempt finished.")
