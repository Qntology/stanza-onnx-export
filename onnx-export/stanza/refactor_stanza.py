import os

STANZA_DIR = r"C:\Users\HP\Desktop\stanza\stanza-main\stanza"

def refactor_stanza_imports():
    old_import = "from torch.nn.utils.rnn import pad_sequence"
    new_import = "from stanza.models.common.traceable_utils import pad_sequence"
    
    count = 0
    for root, dirs, files in os.walk(STANZA_DIR):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                if old_import in content:
                    print(f"  🔧 Patching {path}")
                    content = content.replace(old_import, new_import)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    count += 1
    print(f"✅ Patched {count} files.")

if __name__ == "__main__":
    refactor_stanza_imports()
