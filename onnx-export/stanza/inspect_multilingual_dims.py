import stanza
import torch

def inspect_lang_dims(lang):
    print(f"\n--- Inspecting Dimensions for: {lang} ---")
    nlp = stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse', download_method=None, device='cpu')
    
    # POS Model
    pos_model = nlp.processors['pos'].trainer.model
    print(f"[POS Model: {lang}]")
    print(f"  Args: {pos_model.args}")
    # Calculate expected input size
    total = 0
    if pos_model.args.get('word_emb_dim', 0) > 0: total += pos_model.args['word_emb_dim']
    if pos_model.args.get('pretrain', False): total += 100 # Default pretrain dim
    if pos_model.args.get('char', False): total += pos_model.args.get('transformed_dim', 125)
    if pos_model.args.get('charlm', False):
        total += pos_model.args.get('charlm_transform_dim', 1024) * 2 # Forward + Backward
    
    print(f"  Calculated Input Size: {total}")
    print(f"  Actual LSTM Input Size: {pos_model.taggerlstm.lstm[0].input_size}")

if __name__ == "__main__":
    for lang in ['en', 'ja', 'zh-hans']:
        inspect_lang_dims(lang)
