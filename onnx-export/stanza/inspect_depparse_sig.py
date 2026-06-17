import stanza
from stanza.models.depparse.model import Depparser
import inspect

print(f"Stanza version: {stanza.__version__}")
sig = inspect.signature(Depparser.forward)
print(f"Depparser.forward signature: {sig}")
print(f"Number of parameters: {len(sig.parameters)}")
