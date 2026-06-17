import torch
import torch.nn as nn
import torch.nn.functional as F
import stanza
import os
import json
import inspect
from stanza.models.pos.model import Tagger
from stanza.models.depparse.model import GraphParser

# --- NUCLEAR PATCH: Directly override class methods ---

def patched_tagger_forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text):
    # This is a simplified, ONNX-friendly forward pass
    # It avoids complex dynamic logic and pad_sequence
    
    # 1. Embeddings
    inputs = []
    if self.args['word_emb_dim'] > 0:
        inputs.append(self.word_emb(word))
    if self.args['pretrain']:
        inputs.append(self.trans_pretrained(self.pretrained_emb(pretrained)))
    if hasattr(self, 'charmodel') and self.charmodel:
        # Assuming charmodel_forward is patched or safe
        char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
        inputs.append(self.trans_char(self.drop(char_reps)))
        
    lstm_inputs = torch.cat(inputs, dim=-1)
    # 2. LSTM (Bypass HighwayLSTM's complex logic if needed, but let's try direct)
    lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens)
    
    # 3. Classifiers
    upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
    upos_pred = self.upos_clf(self.drop(upos_hid))
    
    # Return in format expected by wrapper (preds list)
    return None, [upos_pred, upos_pred, upos_pred] # Dummy duplicates for xpos/ufeats if not needed

# Apply the patch to the class
# Tagger.forward = patched_tagger_forward

# --- REFINED EXPORTER ---

def export_stubborn_lang(lang):
    print(f"\n🚀 Nuclear Processing: {lang}")
    ONNX_DIR = "./onnx_final"
    os.makedirs(ONNX_DIR, exist_ok=True)
    
    try:
        # For stubborn languages, we might want to try even more aggressive nocharlm
        stanza.download(lang, processors='tokenize,pos,lemma,depparse')
        nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse')
        
        pos_model = nlp.processors['pos'].trainer.model
        dep_model = nlp.processors['depparse'].trainer.model
        
        # Check if they use charlm
        if hasattr(pos_model, 'charmodel') and pos_model.charmodel:
             print(f"  {lang} uses charmodel, applying extra caution...")

        # ... (rest of export logic similar to definitive v5 but with class-level patches)
        # Actually, let's try a simpler fix first: just use Opset 15 or 17?
        # No, Opset 14 is the most stable for many runtimes.
        
        print(f"✅ Still in development for {lang} nuclear mode.")
    except Exception as e:
        print(f"❌ Failed: {lang} - {e}")

if __name__ == "__main__":
    export_stubborn_lang('ja')
