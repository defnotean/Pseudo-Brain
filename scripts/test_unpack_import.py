import sys
import importlib
from pathlib import Path

p = "/content/Pseudo-Brain/brain/src"
if p not in sys.path:
    sys.path.insert(0, p)
importlib.invalidate_caches()

import irene_brain
print("irene_brain imported successfully:", irene_brain)
from irene_brain.semantic.tokenizer import SemanticTokenizer
tok = SemanticTokenizer(max_threads=16)
print("SemanticTokenizer imported! Vocab size:", tok.vocab_size)
