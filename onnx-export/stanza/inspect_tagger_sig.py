import stanza
from stanza.models.pos.model import Tagger
import inspect

print(f"Stanza version: {stanza.__version__}")
sig = inspect.signature(Tagger.forward)
print(f"Tagger.forward signature: {sig}")
print(f"Number of parameters: {len(sig.parameters)}")
