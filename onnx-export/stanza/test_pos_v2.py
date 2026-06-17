import stanza
import torch
import torch.nn as nn

# Load model
model_dir = './temp_models'
nlp = stanza.Pipeline(lang='ko', dir=model_dir, processors='tokenize,pos,lemma,depparse', verbose=True, use_gpu=False)
pos_model = nlp.processors['pos'].trainer.model
pos_model.eval()

# Create Wrapper to eliminate 'text' dependency from traced graph
class POSTracingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, word, word_mask, wordchars, wordchars_mask, dummy_upos, dummy_xpos, dummy_ufeats, pretrained, word_orig_idx, sentlens, wordlens):
        # We handle text here, not as an input
        text = [['word'] * word.size(1)]
        # Pass None for label inputs to ensure inference mode
        return self.model(word, word_mask, wordchars, wordchars_mask, None, None, None, pretrained, word_orig_idx, sentlens, wordlens, text)

# Initialize Wrapper
pos_wrapper = POSTracingWrapper(pos_model)

# Prepare inputs
seq_len = 32
batch_size = 1
word = torch.randint(1, 100, (batch_size, seq_len), dtype=torch.long)
word_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
wordchars = torch.randint(1, 100, (batch_size * seq_len, 5), dtype=torch.long)
wordchars_mask = torch.ones(batch_size * seq_len, 5, dtype=torch.long)
# Dummy tensors for ONNX tracing
dummy_upos = torch.zeros((batch_size, seq_len), dtype=torch.long)
dummy_xpos = torch.zeros((batch_size, seq_len), dtype=torch.long)
dummy_ufeats = torch.zeros((batch_size, seq_len, 2), dtype=torch.long)

pretrained = torch.randint(1, 100, (batch_size, seq_len), dtype=torch.long)
word_orig_idx = torch.arange(batch_size * seq_len)
sentlens = torch.tensor([seq_len], dtype=torch.long)
wordlens = torch.tensor([5] * (batch_size * seq_len), dtype=torch.long)

print("Exporting POS to ONNX...")
torch.onnx.export(
    pos_wrapper, 
    (word, word_mask, wordchars, wordchars_mask, dummy_upos, dummy_xpos, dummy_ufeats, pretrained, word_orig_idx, sentlens, wordlens),
    "pos_model.onnx",
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'dummy_upos', 'dummy_xpos', 'dummy_ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens'],
    output_names=['loss', 'preds'],
    dynamic_axes={'word': {0: 'batch', 1: 'seq'}, 'word_mask': {0: 'batch', 1: 'seq'}}
)
print("POS model exported successfully.")
