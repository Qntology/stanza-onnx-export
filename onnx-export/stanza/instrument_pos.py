import stanza
import torch
import torch.nn as nn
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Instrumenter")

def register_hooks(model):
    for name, module in model.named_modules():
        def hook(mod, inp, out, name=name):
            logger.info(f"Module: {name}")
            logger.info(f"  Input types: {[type(x) for x in inp] if isinstance(inp, tuple) else type(inp)}")
            # Handle tuple inputs commonly found in RNNs
            if isinstance(inp, tuple):
                for i, x in enumerate(inp):
                    if isinstance(x, torch.Tensor):
                        logger.info(f"    Input {i} shape: {x.shape}")
            if isinstance(out, torch.Tensor):
                logger.info(f"  Output shape: {out.shape}")
            elif isinstance(out, tuple):
                for i, x in enumerate(out):
                    if isinstance(x, torch.Tensor):
                        logger.info(f"    Output {i} shape: {x.shape}")
        
        module.register_forward_hook(hook)

# Load pipeline and model
model_dir = './temp_models'
nlp = stanza.Pipeline(lang='ko', dir=model_dir, processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=False)
pos_model = nlp.processors['pos'].trainer.model
pos_model.eval()

# Register hooks
register_hooks(pos_model)

# Prepare real-like input
doc = nlp("안녕하세요.")
batch = nlp.processors['pos'].trainer.unpack_batch(
    next(iter(nlp.processors['pos'].trainer.dataset.to_length_limited_loader(batch_size=1))),
    torch.device('cpu')
)[0]

print("Starting instrumental forward pass...")
try:
    pos_model(*batch)
    print("Forward pass completed without error.")
except Exception as e:
    print(f"Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
