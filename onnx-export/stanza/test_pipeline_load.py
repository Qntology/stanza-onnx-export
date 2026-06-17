import stanza
import os
import torch

try:
    # Set the model directory to where we extracted the models
    model_dir = './temp_models'
    
    # Try loading a pipeline with only tokenize, pos, lemma and depparse
    print("Attempting to load pipeline...")
    nlp = stanza.Pipeline(lang='ko', dir=model_dir, processors='tokenize,pos,lemma,depparse', verbose=True, use_gpu=False)
    
    print("Pipeline loaded successfully.")
    
    # Extract POS model
    pos_processor = nlp.processors['pos']
    pos_model = pos_processor.trainer.model
    print(f"POS model type: {type(pos_model)}")
    
    # Extract Depparse model
    depparse_processor = nlp.processors['depparse']
    depparse_model = depparse_processor.trainer.model
    print(f"Depparse model type: {type(depparse_model)}")

except Exception as e:
    print(f"Failed: {e}")
