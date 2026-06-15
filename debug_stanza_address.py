import stanza
import json
import os
from config import STANZA_RESOURCES_DIR

def analyze_address_with_stanza():
    samples = {
        "ko": "서울특별시 서초구 반포대로 23길 45 (삼성동)",
        "en": "1234 Market St, San Francisco, CA 94103, USA"
    }

    results = {}

    for lang, text in samples.items():
        print(f"--- Analyzing {lang} address ---")
        try:
            # POS 프로세서를 포함하여 정밀 분석
            nlp = stanza.Pipeline(
                lang=lang,
                model_dir=STANZA_RESOURCES_DIR,
                processors='tokenize,pos',
                use_gpu=False
            )
            doc = nlp(text)
            
            analysis = []
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    for word in token.words:
                        analysis.append({
                            "text": word.text,
                            "lemma": word.lemma if hasattr(word, 'lemma') else None,
                            "upos": word.upos,
                            "xpos": word.xpos,
                            "feats": word.feats if hasattr(word, 'feats') else None
                        })
            results[lang] = analysis
        except Exception as e:
            results[lang] = f"Error: {str(e)}"

    output_path = "stanza_address_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nAnalysis saved to {output_path}")

if __name__ == "__main__":
    analyze_address_with_stanza()
