import importlib
import sys
from pathlib import Path
import torch

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
print(f"Running on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

tokenizer = SemanticTokenizer(max_threads=16)

ckpt_path = Path("/content/tier2_conversational_champion.pt")
print(f"Loading checkpoint {ckpt_path}...")
ckpt = torch.load(str(ckpt_path), map_location=device)
cfg = ckpt.get("config", {})

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
print(f"Model loaded successfully! Parameters: {sum(p.numel() for p in model.parameters()):,}")

sess = StreamingCognitiveSession(model=model, tokenizer=tokenizer, device=device)

regular_person_prompts = [
    "Hello! How are you doing today?",
    "Who are you and what is your name?",
    "Can you talk like a regular person?",
    "I'm playing games right now.",
    "I just won my match in the lobby with my team!",
    "Tell me a joke.",
    "Why is the sky blue?",
    "What should I eat for dinner?",
    "Remember Alice likes coffee.",
    "What does Alice like?",
    "Tell me something cool.",
    "Goodbye, talk to you later!"
]

print("\n" + "=" * 80)
print("TALKING TO PSEUDO-BRAIN LIKE A REGULAR PERSON (Step = %s)" % ckpt.get("step"))
print("=" * 80)

for p in regular_person_prompts:
    res = sess.generate_response(prompt_text=p, thread_id=0, max_new_tokens=40, temperature=0.3, repetition_penalty=1.15)
    ans = res["response_text"].strip()
    print(f"\nUser:  {p}")
    print(f"Brain: {ans}")
    print(f"       [Latency: {res['mean_step_latency_ms']:.2f} ms/step]")

print("\n" + "=" * 80)
