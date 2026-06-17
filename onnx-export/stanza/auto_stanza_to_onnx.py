import os
import requests
import zipfile
import glob
import torch
import shutil
import argparse
import logging
import json
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.foundation_cache import FoundationCache
from stanza.models.tokenization.trainer import Trainer as TokenizerTrainer
from stanza.models.mwt.trainer import Trainer as MWTTrainer
from stanza.models.pos.trainer import Trainer as POSTrainer
from stanza.models.depparse.trainer import Trainer as DepparseTrainer

def setup_logging(lang):
    log_file = f"conversion_{lang}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_pretrain(pt_path):
    base_dir = os.path.dirname(os.path.dirname(pt_path))
    pretrain_path = os.path.join(base_dir, "pretrain", "conll17.pt")
    if os.path.exists(pretrain_path):
        return Pretrain(filename=pretrain_path)
    return None

def download_and_extract_stanza_models(lang, output_dir):
    url = f"https://huggingface.co/stanfordnlp/stanza-{lang}/resolve/main/models/default.zip"
    zip_path = f"{lang}_models.zip"
    extract_path = os.path.join(output_dir, lang)
    
    logging.info(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            logging.info("Download complete. Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            logging.info(f"Extracted to {extract_path}")
            return extract_path
        else:
            logging.error(f"Failed to download. Status code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Download error: {e}")
        return None

def export_tokenizer(model, onnx_path):
    model.eval()
    seq_len = 128
    dummy_input = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
    try:
        emb_dim = model.embeddings.embedding_dim
        rnn_input_dim = model.rnn.input_size
        feat_dim = rnn_input_dim - emb_dim
        dummy_feats = torch.zeros(1, seq_len, feat_dim, dtype=torch.float)
        dummy_lengths = torch.tensor([seq_len], dtype=torch.long)
        
        torch.onnx.export(
            model, (dummy_input, dummy_feats, dummy_lengths), onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['input', 'feats', 'lengths'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 1: 'seq'}, 'feats': {0: 'batch_size', 1: 'seq'}, 'lengths': {0: 'batch_size'}, 'output': {0: 'batch_size', 1: 'seq'}}
        )
        return True
    except Exception as e:
        logging.error(f"Failed to export tokenizer: {e}")
        return False

def export_mwt(model, onnx_path):
    model.eval()
    seq_len = 32
    dummy_src = torch.randint(1, 50, (1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long) 
    
    try:
        torch.onnx.export(
            model, (dummy_src, dummy_mask), onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['src', 'src_mask'], output_names=['logits'],
            dynamic_axes={'src': {0: 'batch_size', 1: 'seq'}, 'src_mask': {0: 'batch_size', 1: 'seq'}, 'logits': {0: 'batch_size', 1: 'seq'}}
        )
        return True
    except Exception as e:
        logging.error(f"Failed to export MWT: {e}")
        return False

def export_pos(model, onnx_path):
    logging.info("Exporting POS model...")
    model.eval()
    # Dummy inputs for POS model (simplified)
    seq_len = 32
    word = torch.randint(1, 100, (1, seq_len), dtype=torch.long)
    word_mask = torch.ones(1, seq_len, dtype=torch.long)
    wordchars = torch.randint(1, 100, (1, seq_len, 5), dtype=torch.long)
    wordchars_mask = torch.ones(1, seq_len, 5, dtype=torch.long)
    upos = None
    xpos = None
    ufeats = None
    pretrained = torch.randint(1, 100, (1, seq_len), dtype=torch.long)
    word_orig_idx = torch.arange(seq_len).unsqueeze(0)
    sentlens = torch.tensor([seq_len], dtype=torch.long)
    wordlens = torch.tensor([seq_len]*seq_len, dtype=torch.long)
    text = [['word'] * seq_len]
    
    # Try exporting
    try:
        # Depending on model architecture, input list might need adjustment
        torch.onnx.export(
            model, (word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, text), 
            onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['word', 'word_mask', 'wordchars', 'wordchars_mask', 'upos', 'xpos', 'ufeats', 'pretrained', 'word_orig_idx', 'sentlens', 'wordlens', 'text'], 
            output_names=['loss', 'preds'],
            # Dynamic axes handling for POS models is very complex due to nested structures
        )
        return True
    except Exception as e:
        logging.error(f"Failed to export POS model: {e}")
        return False

def export_depparse(model, onnx_path):
    logging.info("Exporting Depparse model...")
    # Similar complexity to POS, needs rigorous input definition
    return False

def process_folder(folder_path, lang):
    onnx_output_dir = f'onnx_models_{lang}'
    os.makedirs(onnx_output_dir, exist_ok=True)
    
    pt_files = glob.glob(os.path.join(folder_path, "**/*.pt"), recursive=True)
    logging.info(f"Found {len(pt_files)} .pt files.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    foundation_cache = FoundationCache()
    
    for pt_path in pt_files:
        if "pretrain" in pt_path: continue
        
        file_name = os.path.basename(pt_path)
        logging.info(f"\nProcessing: {pt_path}")
        
        pretrain = load_pretrain(pt_path)
        
        # 1. Tokenizer
        try:
            trainer = TokenizerTrainer(model_file=pt_path)
            if export_tokenizer(trainer.model, os.path.join(onnx_output_dir, f"{file_name}_tokenize.onnx")):
                logging.info(f"Success: Tokenizer")
                continue
        except Exception as e:
            logging.debug(f"Not a Tokenizer")

        # 2. MWT
        try:
            trainer = MWTTrainer(model_file=pt_path)
            if export_mwt(trainer.model, os.path.join(onnx_output_dir, f"{file_name}_mwt.onnx")):
                logging.info(f"Success: MWT")
                continue
        except Exception as e:
            logging.debug(f"Not an MWT")
        
        # 3. POS
        try:
            trainer = POSTrainer(device=device)
            trainer.load(filename=pt_path, pretrain=pretrain, foundation_cache=foundation_cache)
            if export_pos(trainer.model, os.path.join(onnx_output_dir, f"{file_name}_pos.onnx")):
                logging.info(f"Success: POS")
                continue
        except Exception as e:
            logging.error(f"Failed to load POS trainer: {e}")
        
        # 4. Depparse
        try:
            trainer = DepparseTrainer.load(filename=pt_path, pretrain=pretrain, device=device, foundation_cache=foundation_cache)
            if export_depparse(trainer.model, os.path.join(onnx_output_dir, f"{file_name}_depparse.onnx")):
                logging.info(f"Success: Depparse")
                continue
        except Exception as e:
            logging.error(f"Failed to load Depparse trainer: {e}")
            
        logging.warning(f"Skipping: {file_name} (Unsupported or unknown architecture)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code (e.g., ko, en)")
    args = parser.parse_args()
    
    setup_logging(args.lang)
    extracted_folder = download_and_extract_stanza_models(args.lang, './temp_models')
    if extracted_folder:
        process_folder(extracted_folder, args.lang)
