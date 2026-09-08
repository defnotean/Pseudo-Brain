import importlib
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

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
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession

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

sess = StreamingCognitiveSession(model=model, tokenizer=tokenizer, device=device)

prompts = [
    "Hello! How are you doing today?",
    "Who are you and what is your name?",
    "Why is the sky blue?",
]

for p in prompts:
    print(f"\n=======================================================")
    print(f"PROMPT: {p}")
    sess.state = sess.model.init_state(batch_size=1, device=sess.device)
    # Step prompt tokens
    toks = tokenizer.encode(p, thread_id=0)
    print("Tokens stepped:", [(t, tokenizer.decode([t])) for t in toks])
    for t in toks:
        logits, _ = sess.step_token(t, thread_id=0)
    # Step resp token
    logits, _ = sess.step_token(tokenizer.resp_id, thread_id=0)
    
    print("\nGeneration steps (top 5 choices per step):")
    for step_i in range(15):
        probs = F.softmax(logits, dim=-1)
        top5_p, top5_idx = torch.topk(probs, 5)
        top_toks = [(int(idx), tokenizer.decode([int(idx)]), f"{float(pr)*100:.1f}%") for idx, pr in zip(top5_idx, top5_p)]
        best_tok = int(top5_idx[0])
        print(f"  Step {step_i+1}: chosen={best_tok} ({tokenizer.decode([best_tok])!r}) | Top choices: {top_toks}")
        if best_tok in (tokenizer.eos_id, tokenizer.sep_id):
            print("  --> Hit STOP token!")
            break
        logits, _ = sess.step_token(best_tok, thread_id=0)
