import stanza
import torch
import inspect

lang = 'sq'
try:
    stanza.download(lang, processors='tokenize,pos')
    nlp = stanza.Pipeline(lang, processors='tokenize,pos')
    pos_model = nlp.processors['pos'].trainer.model
    pos_model.cpu().eval()
    
    sig = inspect.signature(pos_model.forward)
    print(f"POS model forward signature: {sig}")
    print(f"Number of parameters: {len(sig.parameters)}")
    
    # Try a dummy call with 12 parameters (excluding self)
    batch_size, seq_len = 1, 5
    inputs = {
        'word': torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long),
        'word_mask': torch.ones((batch_size, seq_len), dtype=torch.bool),
        'wordchars': torch.randint(0, 10, (batch_size * seq_len, 10), dtype=torch.long),
        'wordchars_mask': torch.ones((batch_size * seq_len, 10), dtype=torch.bool),
        'upos': None,
        'xpos': None,
        'ufeats': None,
        'pretrained': torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long),
        'word_orig_idx': torch.arange(batch_size * seq_len, dtype=torch.long),
        'sentlens': torch.tensor([seq_len] * batch_size, dtype=torch.long),
        'wordlens': torch.ones((batch_size * seq_len,), dtype=torch.long),
        'text': None
    }
    
    print("Attempting call with 12 parameters...")
    try:
        outputs = pos_model(**inputs)
        print("Success!")
    except Exception as e:
        print(f"Failed with **inputs: {e}")
        
    print("Attempting call with positional arguments...")
    try:
        args = [inputs[name] for name in ['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens', 'text']]
        outputs = pos_model(*args)
        print("Success!")
    except Exception as e:
        print(f"Failed with *args: {e}")

except Exception as e:
    print(f"Error: {e}")
