import importlib
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

REPO_DIR = Path("/content/Pseudo-Brain")
src_path = str(REPO_DIR / "brain" / "src")
exp_path = str(REPO_DIR / "brain" / "experiments")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if exp_path not in sys.path:
    sys.path.insert(0, exp_path)
importlib.invalidate_caches()

from irene_brain.semantic.tokenizer import SemanticTokenizer
from irene_brain.semantic.native_semantic_model import make_semantic_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = SemanticTokenizer(max_threads=16)

ckpt = torch.load("/content/tier2_conversational_champion.pt", map_location=device)
cfg = ckpt["config"]
model = make_semantic_model(
    model_type="pseudo_brain_tier2",
    vocab_size=cfg.get("vocab_size", 2048),
    K=cfg.get("K", 16),
    proj_dim=cfg.get("proj_dim", 4096),
    rank=cfg.get("rank", 32),
    num_deep_layers=cfg.get("num_deep_layers", 2),
    conditional_recurrence=True,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

CURATED = [
    ("Hello!", "Hello there! How is your day going?"),
    ("Good morning gamers, how are you today?", "Good morning gamers! I am doing great and ready to chat. How are you doing today?"),
    ("I'm playing games right now.", "Awesome! What game are you playing? Hope you're winning!"),
    ("I just won my match in the lobby with my team!", "Awesome clutch! Teamwork makes the dream work. How close was the match?"),
    ("Who are you?", "I am Pseudo-Brain, an autonomous recurrent cognitive agent designed to think and converse like a human."),
    ("What is your name?", "My name is Pseudo-Brain! You can think of me as your AI friend and companion."),
    ("Can you talk like a regular person?", "Yes, absolutely! I want to talk just like a normal friend so we can chat about anything easily."),
    ("Why is the sky blue?", "The sky is blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more than red wavelengths."),
    ("Remember Alice likes coffee.", "Noted: Alice prefers coffee."),
    ("What does Alice like?", "Alice likes coffee."),
]

total_loss = 0.0
print(f"=== CHECKPOINT EVALUATION (Step {ckpt.get('step')}) ===")
for p, resp in CURATED:
    turn_str = f"[THREAD:0]{p} [RESP]{resp}[EOS]"
    toks = tokenizer.encode(turn_str)
    resp_id = tokenizer.resp_id
    targets = [-100] * len(toks)
    r_idx = toks.index(resp_id)
    for ti in range(r_idx, len(toks) - 1):
        targets[ti] = toks[ti + 1]

    t_toks = torch.tensor([toks], device=device)
    t_targs = torch.tensor([targets], device=device)
    t_thrd = torch.zeros_like(t_toks)

    with torch.no_grad():
        logits = model(t_toks, thread_seq=t_thrd)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), t_targs.view(-1), ignore_index=-100)
    
    print(f"Loss: {loss.item():.4f} | Prompt: {p}")
    total_loss += loss.item()

print(f"\nMean Curated Loss: {total_loss / len(CURATED):.4f}")
