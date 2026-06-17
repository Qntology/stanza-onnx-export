import stanza
import torch

def inspect_depparse_sizes(lang):
    print(f"\n--- Inspecting Depparse for {lang} ---")
    nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse', download_method=None, device='cpu')
    m = nlp.processors['depparse'].trainer.model
    
    print(f"Args:")
    for k in ['word_emb_dim', 'pretrain', 'tag_emb_dim', 'char_emb_dim', 'char', 'use_upos', 'use_xpos', 'use_ufeats']:
        print(f"  {k}: {m.args.get(k)}")
        
    print("\nLayer dims:")
    if hasattr(m, 'word_emb'): print(f"  word_emb: {m.word_emb.embedding_dim}")
    if hasattr(m, 'pretrained_emb'): print(f"  pretrained_emb: {m.pretrained_emb.weight.shape[1]}")
    if hasattr(m, 'lemma_emb'): print(f"  lemma_emb: {m.lemma_emb.embedding_dim}")
    if hasattr(m, 'upos_emb'): print(f"  upos_emb: {m.upos_emb.embedding_dim}")
    
    if hasattr(m, 'xpos_emb'):
        if isinstance(m.xpos_emb, torch.nn.ModuleList):
             print(f"  xpos_emb (Composite): {[e.embedding_dim for e in m.xpos_emb]}")
        else:
             print(f"  xpos_emb: {m.xpos_emb.embedding_dim}")
             
    if hasattr(m, 'ufeats_emb'):
        if isinstance(m.ufeats_emb, torch.nn.ModuleList):
             print(f"  ufeats_emb (Composite): {[e.embedding_dim for e in m.ufeats_emb]}")
        else:
             print(f"  ufeats_emb: {m.ufeats_emb.embedding_dim}")

    if hasattr(m, 'charmodel'):
        print(f"  char_rep_dim: {m.args['char_emb_dim']}") # trans_char likely output size

    print(f"\nLSTM Input Size: {m.parserlstm.lstm[0].lstm.input_size}")

if __name__ == "__main__":
    inspect_depparse_sizes('ko')
