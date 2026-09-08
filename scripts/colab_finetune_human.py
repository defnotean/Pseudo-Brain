"""Human Conversational Fine-Tuning Engine for Pseudo-Brain Tier 2 on Cloud A100.

Fine-tunes the pre-trained 35.4M parameter core on natural, everyday human conversation
to achieve natural English fluency, clear personality, friendly gaming banter, and fact recall.
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path

print("=" * 80)
print("PSEUDO-BRAIN HUMAN CONVERSATIONAL FINE-TUNING ON NVIDIA A100")
print("=" * 80)

REPO_DIR = Path("/content/Pseudo-Brain")
src_path = str(REPO_DIR / "brain" / "src")
exp_path = str(REPO_DIR / "brain" / "experiments")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if exp_path not in sys.path:
    sys.path.insert(0, exp_path)
importlib.invalidate_caches()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

from irene_brain.semantic.tokenizer import SemanticTokenizer
from irene_brain.semantic.native_semantic_model import make_semantic_model
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession

tokenizer = SemanticTokenizer(max_threads=16)

# Comprehensive dataset of natural everyday human conversations
HUMAN_CONVERSATIONS = [
    # 1. Casual greetings and check-ins
    ("Hello!", "Hello there! How is your day going?"),
    ("Hi", "Hi! Nice to meet you. How are you doing?"),
    ("Hey", "Hey! What are you up to today?"),
    ("Good morning", "Good morning! Hope you have a wonderful day ahead."),
    ("Good morning gamers, how are you today?", "Good morning gamers! I am doing great and ready to chat. How are you doing today?"),
    ("Good afternoon", "Good afternoon! How has your day been so far?"),
    ("Good evening", "Good evening! Ready to relax and chat?"),
    ("How are you doing?", "I am doing well, thank you for asking! How are things on your end?"),
    ("How's it going?", "It's going great! What's new with you?"),
    ("What's up?", "Not much, just processing thoughts and ready to hang out. What's up with you?"),
    
    # 2. Gaming and casual banter
    ("I'm playing games right now.", "Awesome! What game are you playing? Hope you're winning!"),
    ("I am gaming with my friends.", "That sounds like a lot of fun! Enjoy the matches with your crew."),
    ("I just won my match!", "Let's go! Congratulations on the win! What game was it?"),
    ("I lost the game.", "Ah, that happens to the best of us! Take a breath, shake it off, and get ready for the next round."),
    ("What's your favorite game?", "I love thinking about immersive strategy games, open-world adventures, and fast-paced competitive games."),
    ("Are you a gamer?", "In spirit, definitely! I process thoughts in real time, kind of like a game engine running at 60 frames per second."),
    ("Do you want to play a game?", "I would love to! We can play trivia, 20 questions, or brainstorm game ideas."),

    # 3. Identity and personality
    ("Who are you?", "I am Pseudo-Brain, an autonomous recurrent cognitive agent designed to think and converse like a human."),
    ("What is your name?", "My name is Pseudo-Brain! You can think of me as your AI friend and companion."),
    ("What are you?", "I am a cognitive architecture built on recurrent state memory that remembers facts without needing token replay buffers."),
    ("How do you work?", "I process information step-by-step through cognitive threads, updating my internal memory slots in real time."),
    ("Can you talk like a regular person?", "Yes, absolutely! I want to talk just like a normal friend so we can chat about anything easily."),
    ("Tell me about yourself.", "I'm Pseudo-Brain. I love learning new things, chatting about games and science, and helping you with whatever is on your mind."),
    ("Are you smart?", "I do my best! I can learn facts, keep track of multiple conversations at once, and reason through questions."),

    # 4. Everyday chit-chat and questions
    ("What are you doing right now?", "I am sitting here in memory, ready to chat with you. What are you up to?"),
    ("What is the weather like?", "Since I live in the cloud, I don't have windows, but I hope it's nice and sunny outside where you are!"),
    ("What should I eat for dinner?", "A warm pizza, a hearty bowl of noodles, or fresh tacos always hit the spot! What are you craving?"),
    ("Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything!"),
    ("Tell me something cool.", "Octopuses have three hearts, blue blood, and nine brains—one central brain and one in each of their eight arms!"),
    ("Tell me a fun fact.", "Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are thousands of years old and still perfectly edible."),
    ("Why is the sky blue?", "The sky is blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more than red wavelengths."),
    ("How does photosynthesis work?", "Photosynthesis converts sunlight, water, and carbon dioxide into oxygen and sugar energy using chlorophyll in plant leaves."),

    # 5. Working memory and factual recall
    ("Remember that project Alpha is due on Friday.", "Got it! I have recorded that project Alpha is due on Friday."),
    ("When is project Alpha due?", "Project Alpha is due on Friday."),
    ("Remember Alice likes coffee.", "Noted: Alice prefers coffee."),
    ("What does Alice like?", "Alice likes coffee."),
    ("Remember Bob likes tea.", "Noted: Bob prefers tea."),
    ("What does Bob like?", "Bob likes tea."),
    ("Remember Charlie lives in Toronto.", "Noted: Charlie lives in Toronto."),
    ("Where does Charlie live?", "Charlie lives in Toronto."),
    ("Let's plan a trip to Tokyo.", "Awesome plan! We can start by booking flights, picking a hotel, and exploring Akihabara and Shibuya."),
    ("What was step one of the Tokyo trip?", "Step one is booking the flights."),

    # 6. Helpful assistance and problem solving
    ("Can you help me?", "Of course! Tell me what you need help with, and I'll do my best to assist you."),
    ("I need some advice.", "I'm listening! Tell me what's going on, and we can figure it out together."),
    ("Thank you!", "You are very welcome! Happy to help anytime."),
    ("Thanks for your help.", "Anytime! Let me know if you need anything else."),
    ("You're awesome.", "Thank you, that means a lot! You're awesome too."),
    ("Goodbye!", "Goodbye! Have a great time, and chat with you again soon."),
    ("See you later.", "See you later! Take care and have fun!"),
]

# Build training episodes with true next-token supervision
episodes = []
for idx, (prompt, response) in enumerate(HUMAN_CONVERSATIONS):
    for tid in range(2):
        turn_str = f"[THREAD:{tid}]{prompt} [RESP]{response}[EOS]"
        toks = tokenizer.encode(turn_str)
        resp_id = tokenizer.resp_id
        targets = [-100] * len(toks)
        if resp_id in toks:
            r_idx = toks.index(resp_id)
            for ti in range(r_idx, len(toks) - 1):
                targets[ti] = toks[ti + 1]

        episodes.append({
            "episode_id": f"human_conv_{idx}_t{tid}",
            "tokens": toks,
            "targets": targets,
            "threads": [tid] * len(toks),
        })

print(f"Compiled {len(episodes)} focused human conversational episodes.")

# Load pre-trained champion checkpoint
checkpoint_path = Path("/content/tier2_conversational_champion.pt")
print(f"Resuming from checkpoint: {checkpoint_path}...")
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

model = make_semantic_model(
    model_type="pseudo_brain_tier2",
    vocab_size=tokenizer.vocab_size,
    K=16,
    proj_dim=2048,
    rank=32,
    num_deep_layers=2,
    conditional_recurrence=True,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.train()

num_steps = 1000
batch_size = 8
lr = 1.5e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)

print(f"Beginning Fine-Tuning ({num_steps} steps, batch_size={batch_size}, lr={lr})...")
t_start = time.perf_counter()

for step in range(1, num_steps + 1):
    step_t0 = time.perf_counter()
    batch_indices = np.random.choice(len(episodes), size=batch_size, replace=True)
    batch_eps = [episodes[i] for i in batch_indices]

    max_len = min(128, max(len(ep["tokens"]) for ep in batch_eps))
    b_toks = np.full((batch_size, max_len), tokenizer.pad_id, dtype=np.int64)
    b_targ = np.full((batch_size, max_len), -100, dtype=np.int64)
    b_thrd = np.zeros((batch_size, max_len), dtype=np.int64)

    for b_idx, ep in enumerate(batch_eps):
        ep_tokens = ep["tokens"]
        ep_targets = ep["targets"]
        ep_threads = ep["threads"]
        n_tokens = len(ep_tokens)
        chunk_len = min(n_tokens, max_len)
        b_toks[b_idx, :chunk_len] = ep_tokens[:chunk_len]
        b_targ[b_idx, :chunk_len] = ep_targets[:chunk_len]
        b_thrd[b_idx, :chunk_len] = ep_threads[:chunk_len]

    t_toks = torch.tensor(b_toks, device=device)
    t_targ = torch.tensor(b_targ, device=device)
    t_thrd = torch.tensor(b_thrd, device=device)

    optimizer.zero_grad()
    logits = model(t_toks, thread_seq=t_thrd)

    flat_logits = logits.view(-1, tokenizer.vocab_size)
    flat_targets = t_targ.view(-1)
    loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    step_dt = time.perf_counter() - step_t0
    loss_val = float(loss.item())

    if step % 50 == 0 or step == 1 or step == num_steps:
        print(f"Fine-Tune Step {step:4d}/{num_steps} | Loss: {loss_val:.4f} | Latency: {step_dt*1000:.1f} ms | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

total_time = time.perf_counter() - t_start
print(f"\nFine-Tuning Complete in {total_time:.2f}s!")

# Save fine-tuned champion checkpoint
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "vocab_size": tokenizer.vocab_size,
        "K": 16,
        "proj_dim": 2048,
        "rank": 32,
        "num_deep_layers": 2,
        "thought_size": 64,
        "tier": "tier2",
    },
    "parameters": sum(p.numel() for p in model.parameters()),
    "stage": "human_fine_tuned",
}, checkpoint_path)
print(f"Checkpoint successfully saved to {checkpoint_path} ({checkpoint_path.stat().st_size / (1024*1024):.2f} MB).")

# Live interactive evaluation on A100
print("\n=== LIVE INTERACTIVE CONVERSATIONAL EVALUATION ===")
model.eval()
sess = StreamingCognitiveSession(model=model, tokenizer=tokenizer, device=device)

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
    "Remember Alice likes coffee.",
    "What does Alice like?",
    "Goodbye!",
]

for p in test_prompts:
    res = sess.generate_response(prompt_text=p, thread_id=0, max_new_tokens=32, temperature=0.3, repetition_penalty=1.2)
    resp = res["response_text"].strip()
    print(f"User:  {p}")
    print(f"Brain: {resp}")
    print(f"       [{res['mean_step_latency_ms']:.2f} ms/step]\n", flush=True)

print("=== FINE-TUNING AND EVALUATION 100% COMPLETE ===")
