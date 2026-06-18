import os
from model_manager import model_manager
from processor import process_pii
import torch

def test_auto_download():
    print("=== Auto-Download & Process Test (French) ===")
    
    # Ensure LLM is loaded
    success, msg = model_manager.load_llm()
    if not success:
        print(f"Failed to load LLM: {msg}")
        return

    # Test French text (should trigger auto-download of 'fr')
    french_text = "Bonjour, je m'appelle Amélie Poulain. Mon numéro est 01 23 45 67 89 et mon email est amelie@example.com."
    
    print(f"\n[Input]: {french_text}")
    print("Processing... (This might take a moment if it needs to download the model)")
    
    masked_text, pos_info, entities, dl_status = process_pii(french_text)
    
    print(f"\n[Auto-Download Status]: {dl_status}")
    print(f"[Masked Output]: {masked_text}")
    print(f"[Detected Entities]: {entities}")
    print(f"[Language/POS Info]:\n{pos_info}")

if __name__ == "__main__":
    test_auto_download()
