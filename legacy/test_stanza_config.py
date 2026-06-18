from model_manager import model_manager
from config import LANG_REGISTRY
import stanza

def test_stanza_configs():
    print("=== Stanza Language Configuration Test ===")
    
    results = []
    for code, info in LANG_REGISTRY.items():
        print(f"\n[Test] {info['name']} ({code})")
        
        # Load stanza with registry processors
        nlp = model_manager.load_stanza(code)
        
        if nlp is not None:
            print(f"✅ Success: {info['name']} loaded.")
            # Verify if MWT was automatically removed if needed
            actual_processors = list(nlp.processors.keys())
            print(f"   Actual Processors: {actual_processors}")
            results.append({"lang": code, "status": "Success", "processors": actual_processors})
        else:
            print(f"❌ Failed: {info['name']} could not be loaded.")
            results.append({"lang": code, "status": "Failed"})
        
        # Unload to save memory
        model_manager.unload_stanza()

    print("\n--- Summary ---")
    for res in results:
        print(f"{res['lang']}: {res['status']} ({res.get('processors', 'N/A')})")

if __name__ == "__main__":
    test_stanza_configs()
