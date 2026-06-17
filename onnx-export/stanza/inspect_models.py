import torch
import os
from stanza.models.pos.trainer import Trainer as POSTrainer
from stanza.models.depparse.trainer import Trainer as DepparseTrainer
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.foundation_cache import FoundationCache

def inspect_model(pt_path, trainer_class, pretrain_path):
    print(f"\n--- Inspecting {pt_path} ---")
    try:
        pretrain = Pretrain(filename=pretrain_path)
        foundation_cache = FoundationCache()
        
        if trainer_class == POSTrainer:
            trainer = POSTrainer(device='cpu')
            trainer.load(filename=pt_path, pretrain=pretrain, foundation_cache=foundation_cache)
        else:
            trainer = DepparseTrainer.load(filename=pt_path, pretrain=pretrain, device='cpu', foundation_cache=foundation_cache)
            
        print("Args:")
        for k, v in trainer.args.items():
            if v is not None and not isinstance(v, (dict, list)):
                print(f"  {k}: {v}")
        
        model = trainer.model
        print("\nModel structure (top-level):")
        for name, child in model.named_children():
            print(f"  {name}: {type(child)}")
            
        # Check if charlm is present
        has_charlm = trainer.args.get('charlm', False)
        print(f"  Has CharLM: {has_charlm}")
        
        # Check if bert is present
        has_bert = trainer.args.get('bert_model', None) is not None
        print(f"  Has BERT: {has_bert}")

    except Exception as e:
        import traceback
        print(f"Error inspecting {pt_path}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    pretrain_path = "./temp_stanza_models/ko/pretrain/conll17.pt"
    inspect_model("./temp_stanza_models/ko/pos/kaist_nocharlm.pt", POSTrainer, pretrain_path)
    inspect_model("./temp_stanza_models/ko/depparse/kaist_nocharlm.pt", DepparseTrainer, pretrain_path)
