import torch
import os
import stanza
from stanza.models.pos.trainer import Trainer as POSTrainer
from stanza.models.depparse.trainer import Trainer as DepparseTrainer
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.common.vocab import CompositeVocab

def inspect_model_detailed(lang):
    print(f"\n--- Detailed Inspection for {lang} ---")
    try:
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
        
        for proc_name in ['pos', 'depparse']:
            print(f"\nProcessor: {proc_name}")
            processor = nlp.processors[proc_name]
            trainer = processor.trainer
            model = trainer.model
            vocab = trainer.vocab
            
            print(f"  Vocab keys: {vocab.keys()}")
            
            if 'xpos' in vocab:
                if isinstance(vocab['xpos'], CompositeVocab):
                    print(f"  XPOS is Composite: {vocab['xpos'].lens()}")
                else:
                    print(f"  XPOS is Simple: {len(vocab['xpos'])}")
            
            if 'feats' in vocab:
                if isinstance(vocab['feats'], CompositeVocab):
                    print(f"  Feats is Composite: {vocab['feats'].lens()}")
            
            # Check model input expectations
            if proc_name == 'pos':
                # Check charmodel
                if hasattr(model, 'charmodel'):
                    print(f"  Has CharacterModel: {type(model.charmodel)}")
            
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    inspect_model_detailed('ko')
