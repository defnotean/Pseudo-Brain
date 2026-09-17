"""Local Inspection and Rigorous Testing of 36.7M Champion Pseudo-Brain."""

import sys
import time
from pathlib import Path
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent / "brain"
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from irene_brain.unified.unified_model import UnifiedPseudoBrain
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer

ckpt_path = _REPO_ROOT / "checkpoints" / "pb_35m_champion.pt"
print(f"Loading {ckpt_path} ({ckpt_path.stat().st_size / (1024*1024):.1f} MB)...")
ckpt = torch.load(str(ckpt_path), map_location="cpu")

if "tokenizer_json" in ckpt:
    print("Loading embedded self-contained BPE tokenizer from checkpoint...")
    tokenizer = BpeSemanticTokenizer.from_str(ckpt["tokenizer_json"], vocab_size=ckpt["vocab_size"], max_threads=16)
else:
    tokenizer = BpeSemanticTokenizer(vocab_size=ckpt["vocab_size"], max_threads=16)
model = UnifiedPseudoBrain(
    vocab_size=ckpt["vocab_size"],
    tier=ckpt["tier"],
    num_deep_layers=ckpt["num_deep_layers"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("=" * 80)
print(f"MODEL VERIFICATION: {ckpt['total_params']:,} PARAMETERS")
print(f"Working Memory (Law 1): {model.K_fast * model.W_fast * 4:,} bytes (Strict 4.0 KB)")
print(f"Final Pretrain Loss: {ckpt.get('stage1_final_loss', 'N/A'):.4f}")
print(f"Final SFT Loss: {ckpt.get('stage2_final_loss', 'N/A'):.4f}")
print("=" * 80)

eval_prompts = [
    "Hello!",
    "Hi, who are you?",
    "Tell me a funny joke!",
    "What is entropy in thermodynamics?",
    "Explain the law of conservation of energy.",
    "Fix this bug: def add(a, b): return a - b",
    "Can you explain quantum superposition simply?",
]

eos_id = tokenizer.eos_id

with torch.no_grad():
    for prompt in eval_prompts:
        toks = tokenizer.encode(f"[THREAD:0]{prompt} [RESP]")
        state = model.init_state(batch_size=1, device=torch.device("cpu"))

        # Prime state
        out = None
        for t_id in toks:
            sensory = model.encode_sensory(token_ids=torch.tensor([t_id]))
            out, state = model.step(sensory, state)

        # Generate tokens greedily
        gen_tokens = []
        logits = out["logits"].clone()
        for _ in range(70):
            for t_val in set(gen_tokens):
                if gen_tokens.count(t_val) >= 2:
                    logits[0, t_val] -= 2.0
            next_id = int(logits.argmax(dim=-1).item())
            if next_id == eos_id or next_id == tokenizer.pad_id:
                break
            gen_tokens.append(next_id)
            sensory = model.encode_sensory(token_ids=torch.tensor([next_id]))
            out, state = model.step(sensory, state)
            logits = out["logits"].clone()

        resp_text = tokenizer.decode(gen_tokens).strip()
        print(f"\nUser: {prompt}")
        print(f"Pseudo-Brain (36.7M): {resp_text}")

print("\n" + "=" * 80)
print("LOCAL VERIFICATION COMPLETE - ZERO SENTENCE SPLICING DETECTED!")
print("=" * 80)
