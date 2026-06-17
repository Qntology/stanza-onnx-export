import stanza
import torch
import torch.nn as nn

def inspect_model():
    print("--- Stanza Model Deep Inspection ---")
    nlp = stanza.Pipeline('ko', processors='tokenize,pos,lemma,depparse', download_method=None, device='cpu')
    
    # POS Model
    pos_model = nlp.processors['pos'].trainer.model
    print(f"\n[POS Model]")
    print(f"Type: {type(pos_model)}")
    for name, child in pos_model.named_children():
        print(f"  Child: {name}, Type: {type(child)}")
    
    if hasattr(pos_model, 'pretrained_emb'):
        print(f"  pretrained_emb type: {type(pos_model.pretrained_emb)}")
        if hasattr(pos_model.pretrained_emb, 'weight'):
            print(f"  pretrained_emb weight shape: {pos_model.pretrained_emb.weight.shape}")
        else:
            print(f"  pretrained_emb has NO weight attribute!")

    # Depparse Model
    dep_model = nlp.processors['depparse'].trainer.model
    print(f"\n[Depparse Model]")
    print(f"Type: {type(dep_model)}")
    for name, child in dep_model.named_children():
        print(f"  Child: {name}, Type: {type(child)}")
    
    if hasattr(dep_model, 'pretrained_emb'):
        print(f"  pretrained_emb type: {type(dep_model.pretrained_emb)}")
        # Check if it's the PretrainedEmbedding wrapper
        from stanza.models.common.pretrain import PretrainedEmbedding
        if isinstance(dep_model.pretrained_emb, PretrainedEmbedding):
            print(f"  It is a PretrainedEmbedding wrapper. Weight shape: {dep_model.pretrained_emb.weight.shape}")

    # Check state_dict keys
    print(f"\n[State Dict Sample Keys]")
    keys = list(dep_model.state_dict().keys())
    for k in keys[:10]:
        print(f"  {k}")

if __name__ == "__main__":
    inspect_model()
