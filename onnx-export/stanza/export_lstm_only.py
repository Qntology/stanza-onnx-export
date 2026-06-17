import stanza
import torch
import torch.nn as nn
from stanza.models.common.hlstm import HighwayLSTM

# Load model
model_dir = './temp_models'
nlp = stanza.Pipeline(lang='ko', dir=model_dir, processors='tokenize,pos,lemma,depparse', verbose=True, use_gpu=False)
pos_model = nlp.processors['pos'].trainer.model
pos_model.eval()

# Extract just the LSTM part
tagger_lstm = pos_model.taggerlstm
tagger_lstm.eval()

# Dummy input for LSTM
# HighwayLSTM input: (batch_size, seq_len, input_size)
batch_size = 1
seq_len = 32
input_size = tagger_lstm.input_size
hidden_dim = tagger_lstm.hidden_size # Corrected
num_layers = tagger_lstm.num_layers

dummy_input = torch.randn(batch_size, seq_len, input_size)
# HighwayLSTM forward(self, input, lengths, hx=None)
lengths = torch.tensor([seq_len], dtype=torch.long)

# Traceable wrapper for HighwayLSTM
class LSTMWrapper(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
        
    def forward(self, x, lengths):
        # We need to simulate the packing here because ONNX doesn't support PackedSequence
        # This is a major hurdle. 
        # Actually, let's try just the raw LSTM call without packing first?
        # No, PackedLSTM needs packed sequences.
        return self.lstm(x, lengths)

print("Exporting LSTM...")
# This will likely fail if PackedLSTM is used.
try:
    torch.onnx.export(
        tagger_lstm, 
        (dummy_input, lengths),
        "lstm_only.onnx",
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input', 'lengths'],
        output_names=['output', 'hidden'],
        dynamic_axes={'input': {0: 'batch', 1: 'seq'}}
    )
    print("LSTM exported successfully.")
except Exception as e:
    print(f"Failed to export LSTM: {e}")
