import stanza
def inspect_lemma(lang):
    nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma', download_method=None, device='cpu')
    m = nlp.processors['lemma'].trainer.model
    print(f"Lemma model type: {type(m)}")
    
if __name__ == "__main__":
    inspect_lemma('ko')
