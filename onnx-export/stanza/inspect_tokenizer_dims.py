import stanza

lang = 'en'
nlp = stanza.Pipeline(lang, processors='tokenize')
tokenizer = nlp.processors['tokenize'].trainer.model

for name, param in tokenizer.named_parameters():
    print(f"{name}: {param.shape}")

print(f"Vocab size (chars): {len(tokenizer.vocab)}")
