import sys
from pathlib import Path
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "brain" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "brain" / "experiments"))

from irene_brain.semantic.tokenizer import SemanticTokenizer
from irene_brain.semantic.native_semantic_model import make_semantic_model
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession

tok = SemanticTokenizer(max_threads=16)
ckpt_path = _REPO_ROOT / "brain" / "checkpoints" / "tier2_conversational_champion.pt"
print(f"Loading checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / (1024*1024):.2f} MB)...")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = make_semantic_model(
    "pseudo_brain_tier2",
    vocab_size=tok.vocab_size,
    K=16,
    proj_dim=4096,
    rank=32,
    num_deep_layers=2,
    conditional_recurrence=True,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

sess = StreamingCognitiveSession(model=model, tokenizer=tok, device=torch.device("cpu"))

# 1. Independent Prompt Evaluation (Fresh Session per prompt)
print("=== PART 1: INDEPENDENT CONVERSATIONAL PROMPTS ===")
test_prompts = [
    "Good morning gamers, how are you today?",
    "Hey, how are you?",
    "Who are you?",
    "What is your name?",
    "I'm playing games right now.",
    "Tell me a joke.",
    "Why is the sky blue?",
    "Remember that project Alpha is due on Friday.",
    "When is project Alpha due?",
]

for p in test_prompts:
    sess.reset()
    res = sess.generate_response(prompt_text=p, thread_id=0, max_new_tokens=32, temperature=0.3, repetition_penalty=1.2)
    resp = res["response_text"].strip()
    print(f"User:  {p}")
    print(f"Brain: {resp}")
    print(f"       [Latency: {res['mean_step_latency_ms']:.2f} ms/step]\n")

# 2. Multi-turn Sequential Conversation (Memory Persistence without token replay)
print("=== PART 2: MULTI-TURN SEQUENTIAL CONVERSATION ===")
sess.reset()
multi_turn = [
    "Hello! What is your name?",
    "Remember Alice likes coffee.",
    "What does Alice like?",
    "Remember that project Alpha is due on Friday.",
    "When is project Alpha due?",
]

for turn in multi_turn:
    res = sess.generate_response(prompt_text=turn, thread_id=0, max_new_tokens=32, temperature=0.3, repetition_penalty=1.2)
    resp = res["response_text"].strip()
    print(f"User:  {turn}")
    print(f"Brain: {resp}")
    print(f"       [Latency: {res['mean_step_latency_ms']:.2f} ms/step]\n")
