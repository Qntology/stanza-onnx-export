import stanza
import inspect

lang = 'en'
nlp = stanza.Pipeline(lang, processors='tokenize')
tokenizer = nlp.processors['tokenize'].trainer.model
print(f"\nEN Tokenizer type: {type(tokenizer)}")
print(f"Forward signature: {inspect.signature(tokenizer.forward)}")
