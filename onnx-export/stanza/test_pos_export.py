import stanza
import os
import torch
import torch.nn as nn

class POSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text):
        # We need to adapt the input to match Tagger.forward
        # For ONNX, we must ensure text is not passed as it's not a tensor
        # Actually, let's see if we can pass text as a dummy
        return self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text)

# ... (rest of the pipeline loading code)

model_dir = './temp_models'
nlp = stanza.Pipeline(lang='ko', dir=model_dir, processors='tokenize,pos,lemma,depparse', verbose=True, use_gpu=False)

pos_model = nlp.processors['pos'].trainer.model
# Create wrapper
pos_wrapper = POSWrapper(pos_model)

# Simplified dummy input for Tagger
# Tagger forward: forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text)

# For ONNX, text is not easily passable. We might need to simplify the POS model's forward method by creating a wrapper that excludes text.

class SimplifiedPOS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, word, word_mask, wordchars, wordchars_mask, pretrained, word_orig_idx, sentlens, wordlens):
        # Setting optional/unused inputs to None
        upos = None
        xpos = None
        ufeats = None
        text = [['word'] * 32] # Text is still an issue, need to see if we can remove it from forward
        return self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text)

# Actually, I should just try to export it as-is first and see the error.
seq_len = 32
batch_size = 1
# Input shapes for POS Tagger
# word: (batch, seq) -> (1, 32)
# wordchars: (batch * seq, char_len) -> (32, 5)
word = torch.randint(1, 100, (batch_size, seq_len), dtype=torch.long)
word_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
wordchars = torch.randint(1, 100, (batch_size * seq_len, 5), dtype=torch.long)
wordchars_mask = torch.ones(batch_size * seq_len, 5, dtype=torch.long)
# Set label inputs to None for inference mode
upos = None
xpos = None
ufeats = None
pretrained = torch.randint(1, 100, (batch_size, seq_len), dtype=torch.long)
word_orig_idx = torch.arange(batch_size * seq_len)
sentlens = torch.tensor([seq_len], dtype=torch.long)
wordlens = torch.tensor([5] * (batch_size * seq_len), dtype=torch.long)
text = [['word'] * seq_len]

print("Exporting POS...")
torch.onnx.export(
    pos_wrapper, 
    (word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text),
    "pos_model.onnx",
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens', 'text'],
    dynamic_axes={'word': {0: 'batch', 1: 'seq'}, 'word_mask': {0: 'batch', 1: 'seq'}} # Added dynamic axes
)

