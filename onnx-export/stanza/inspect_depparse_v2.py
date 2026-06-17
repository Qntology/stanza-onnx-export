import stanza
import torch

def inspect_depparse_layers(lang):
    nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse', download_method=None, device='cpu')
    m = nlp.processors['depparse'].trainer.model
    
    print(f"trans_pretrained: {m.trans_pretrained.in_features} -> {m.trans_pretrained.out_features}")
    if hasattr(m, 'trans_char'):
        print(f"trans_char: {m.trans_char.in_features} -> {m.trans_char.out_features}")
    
    if hasattr(m, 'upos_emb'): print(f"upos_emb: {m.upos_emb.embedding_dim}")
    if hasattr(m, 'xpos_emb'): 
        if isinstance(m.xpos_emb, torch.nn.ModuleList):
            print(f"xpos_emb components: {[e.embedding_dim for e in m.xpos_emb]}")
        else:
            print(f"xpos_emb: {m.xpos_emb.embedding_dim}")
    
    # Check what 'inputs' are being added
    # We can't easily run it, but we can check args
    print(f"LSTM input_size: {m.parserlstm.lstm[0].lstm.input_size}")

if __name__ == "__main__":
    inspect_depparse_layers('ko')
