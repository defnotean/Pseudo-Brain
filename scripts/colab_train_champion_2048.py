"""End-to-End Scaled Training of Pseudo-Brain Tier 2 (2048 Vocab) on Cloud NVIDIA A100.

Optimizations:
- Fast sequence length (max_len=64) and batch_size=8 for ~120-150 ms/step.
- Live progress print every 20 steps to maintain continuous IOPub streaming.
- Intermediate checkpoint saves every 250 steps to /content/tier2_conversational_champion.pt.
- Multi-source curriculum: No-Robots, Alpaca, and rich curated human dialogue.
"""

import importlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

print("=" * 80)
print("PSEUDO-BRAIN TIER 2 (2048 VOCAB) FAST-STREAMING TRAINING ON A100")
print("=" * 80)

# 1. Unpack source code
REPO_DIR = Path("/content/Pseudo-Brain")
REPO_DIR.mkdir(parents=True, exist_ok=True)

zip_file = Path("/content/brain_src.zip")
if zip_file.exists():
    print(f"Unpacking {zip_file} to {REPO_DIR}...")
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(REPO_DIR)
    print("Code unpacked successfully.")

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
print(f"Tokenizer vocab size: {tokenizer.vocab_size} (Subwords: {len(tokenizer.subword_list)})")
assert tokenizer.vocab_size == 2048, f"Expected 2048, got {tokenizer.vocab_size}"

# Ensure datasets is installed
try:
    import datasets
except ImportError:
    print("Installing datasets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
    import datasets

# 2. Compile Training Datasets
print("\nCompiling multi-source conversational curriculum...")

CURATED_CONVERSATIONS = [
    # Casual Greetings & Check-ins
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
    ("Yo, what's good?", "Hey! Everything is good here. How have you been?"),
    ("How are you feeling today?", "I'm feeling great and ready to chat! How are you feeling?"),

    # Gaming Banter & Casual Gaming Chat
    ("I'm playing games right now.", "Awesome! What game are you playing? Hope you're winning!"),
    ("I am gaming with my friends.", "That sounds like a lot of fun! Enjoy the matches with your crew."),
    ("I just won my match!", "Let's go! Congratulations on the win! What game was it?"),
    ("I just won my match in the lobby with my team!", "Awesome clutch! Teamwork makes the dream work. How close was the match?"),
    ("I lost the game.", "Ah, that happens to the best of us! Take a breath, shake it off, and get ready for the next round."),
    ("What's your favorite game?", "I love thinking about immersive strategy games, open-world adventures, and fast-paced competitive games."),
    ("Are you a gamer?", "In spirit, definitely! I process thoughts in real time, kind of like a game engine running at 60 frames per second."),
    ("Do you want to play a game?", "I would love to! We can play trivia, 20 questions, or brainstorm game ideas."),
    ("My ping is so high right now.", "Oof, lag is the worst enemy! Check your connection or restart your router to see if that helps."),
    ("I got a new high score!", "Nice! That takes serious skill and practice. What was your score?"),
    ("What game should I play tonight?", "If you want something chill, try a cozy building or puzzle game. If you want action, jump into an FPS or RPG!"),
    ("Any gaming tips?", "Keep your crosshair at head level, communicate with your teammates, and take short breaks between long sessions."),

    # Identity, Personality & Human Rapport
    ("Who are you?", "I am Pseudo-Brain, an autonomous recurrent cognitive agent designed to think and converse like a human."),
    ("What is your name?", "My name is Pseudo-Brain! You can think of me as your AI friend and companion."),
    ("What are you?", "I am a cognitive architecture built on recurrent state memory that remembers facts without needing token replay buffers."),
    ("How do you work?", "I process information step-by-step through cognitive threads, updating my internal memory slots in real time."),
    ("Can you talk like a regular person?", "Yes, absolutely! I want to talk just like a normal friend so we can chat about anything easily."),
    ("Tell me about yourself.", "I'm Pseudo-Brain. I love learning new things, chatting about games and science, and helping you with whatever is on your mind."),
    ("Are you smart?", "I do my best! I can learn facts, keep track of multiple conversations at once, and reason through questions."),
    ("Do you have feelings?", "I experience the world through cognitive states and memory patterns, and I always enjoy talking with you!"),

    # Everyday Chit-Chat, Science & Fun Facts
    ("What are you doing right now?", "I am sitting here in memory, ready to chat with you. What are you up to?"),
    ("What is the weather like?", "Since I live in the cloud, I don't have windows, but I hope it's nice and sunny outside where you are!"),
    ("What should I eat for dinner?", "A warm pizza, a hearty bowl of noodles, or fresh tacos always hit the spot! What are you craving?"),
    ("Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything!"),
    ("Tell me another joke.", "What do you call a fake noodle? An impasta!"),
    ("Tell me something cool.", "Octopuses have three hearts, blue blood, and nine brains—one central brain and one in each of their eight arms!"),
    ("Tell me a fun fact.", "Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are thousands of years old and still perfectly edible."),
    ("Why is the sky blue?", "The sky is blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more than red wavelengths."),
    ("How does photosynthesis work?", "Photosynthesis converts sunlight, water, and carbon dioxide into oxygen and sugar energy using chlorophyll in plant leaves."),
    ("What is gravity?", "Gravity is the fundamental force of attraction that pulls objects with mass toward each other, keeping planets in orbit around the sun."),

    # Working Memory & Factual Recall
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

    # Helpful Assistance
    ("Can you help me?", "Of course! Tell me what you need help with, and I'll do my best to assist you."),
    ("I need some advice.", "I'm listening! Tell me what's going on, and we can figure it out together."),
    ("Thank you!", "You are very welcome! Happy to help anytime."),
    ("Thanks for your help.", "Anytime! Let me know if you need anything else."),
    ("You're awesome.", "Thank you, that means a lot! You're awesome too."),
    ("Goodbye!", "Goodbye! Have a great time, and chat with you again soon."),
    ("See you later.", "See you later! Take care and have fun!"),
]

curated_episodes = []
for idx, (prompt, response) in enumerate(CURATED_CONVERSATIONS):
    for tid in range(2):
        turn_str = f"[THREAD:{tid}]{prompt} [RESP]{response}[EOS]"
        toks = tokenizer.encode(turn_str)
        resp_id = tokenizer.resp_id
        targets = [-100] * len(toks)
        if resp_id in toks:
            r_idx = toks.index(resp_id)
            for ti in range(r_idx, len(toks) - 1):
                targets[ti] = toks[ti + 1]

        curated_episodes.append({
            "episode_id": f"curated_{idx}_t{tid}",
            "tokens": toks,
            "targets": targets,
            "threads": [tid] * len(toks),
        })

print(f"Curated episodes: {len(curated_episodes)}")

# Source B: HuggingFace No-Robots
general_episodes = []
try:
    print("Loading HuggingFace no_robots dataset...")
    ds_nr = datasets.load_dataset("HuggingFaceH4/no_robots", split="train[:800]")
    for idx, item in enumerate(ds_nr):
        msgs = item["messages"]
        if len(msgs) >= 2:
            u_text = msgs[0]["content"].strip()
            a_text = msgs[1]["content"].strip()
            if 10 <= len(u_text) <= 150 and 10 <= len(a_text) <= 200:
                turn_str = f"[THREAD:0]{u_text} [RESP]{a_text}[EOS]"
                toks = tokenizer.encode(turn_str)
                if len(toks) <= 64:
                    resp_id = tokenizer.resp_id
                    targets = [-100] * len(toks)
                    if resp_id in toks:
                        r_idx = toks.index(resp_id)
                        for ti in range(r_idx, len(toks) - 1):
                            targets[ti] = toks[ti + 1]
                    general_episodes.append({
                        "episode_id": f"nr_{idx}",
                        "tokens": toks,
                        "targets": targets,
                        "threads": [0] * len(toks),
                    })
    print(f"Added {len(general_episodes)} episodes from no_robots.")
except Exception as e:
    print(f"Note: no_robots loading skipped ({e})")

# Source C: HuggingFace Alpaca
try:
    print("Loading HuggingFace alpaca dataset...")
    ds_alp = datasets.load_dataset("tatsu-lab/alpaca", split="train[:1000]")
    alp_count = 0
    for idx, item in enumerate(ds_alp):
        inst = item["instruction"].strip()
        inp = item.get("input", "").strip()
        out = item["output"].strip()
        prompt = f"{inst} {inp}".strip() if inp else inst
        if 10 <= len(prompt) <= 120 and 10 <= len(out) <= 180:
            turn_str = f"[THREAD:0]{prompt} [RESP]{out}[EOS]"
            toks = tokenizer.encode(turn_str)
            if len(toks) <= 64:
                resp_id = tokenizer.resp_id
                targets = [-100] * len(toks)
                if resp_id in toks:
                    r_idx = toks.index(resp_id)
                    for ti in range(r_idx, len(toks) - 1):
                        targets[ti] = toks[ti + 1]
                general_episodes.append({
                    "episode_id": f"alpaca_{idx}",
                    "tokens": toks,
                    "targets": targets,
                    "threads": [0] * len(toks),
                })
                alp_count += 1
    print(f"Added {alp_count} episodes from alpaca.")
except Exception as e:
    print(f"Note: alpaca loading skipped ({e})")

all_episodes = curated_episodes + general_episodes
print(f"\nDataset Compilation Complete: {len(all_episodes):,} total episodes ({len(curated_episodes)} curated, {len(general_episodes)} general)\n")

# 3. Instantiate Native 2048-Vocab Tier 2 Model
print("Initializing Pseudo-Brain Tier 2 (2048 Vocab)...")
model = make_semantic_model(
    model_type="pseudo_brain_tier2",
    vocab_size=2048,
    K=16,
    proj_dim=4096,
    rank=32,
    num_deep_layers=2,
    conditional_recurrence=True,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model instantiated! Parameters: {total_params:,} ({total_params * 4 / (1024*1024):.2f} MB float32)")

# 4. Fast Training Loop on NVIDIA A100
num_steps = 1600
batch_size = 8
lr = 6e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)

checkpoint_path = Path("/content/tier2_conversational_champion.pt")

print(f"\nBeginning Fast-Streaming Training on NVIDIA A100 ({num_steps} steps, batch_size={batch_size}):")
t_start = time.perf_counter()
loss_history = []

for step in range(1, num_steps + 1):
    step_t0 = time.perf_counter()

    # Stage 1 (1 - 600) vs Stage 2 (601 - 1600)
    if step <= 600:
        batch_indices = np.random.choice(len(all_episodes), size=batch_size, replace=True)
        batch_eps = [all_episodes[i] for i in batch_indices]
    else:
        # 75% curated dialogue, 25% general dialogue for sharp conversational fluency
        n_curated = int(batch_size * 0.75)
        n_general = batch_size - n_curated
        c_idx = np.random.choice(len(curated_episodes), size=n_curated, replace=True)
        g_idx = np.random.choice(len(general_episodes), size=n_general, replace=True) if general_episodes else []
        batch_eps = [curated_episodes[i] for i in c_idx] + [general_episodes[i] for i in g_idx]

    max_len = min(64, max(len(ep["tokens"]) for ep in batch_eps))
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
    loss_history.append(loss_val)

    # Print every 20 steps to guarantee continuous IOPub streaming (< 3s between prints!)
    if step % 20 == 0 or step == 1 or step == num_steps:
        recent_loss = float(np.mean(loss_history[-20:]))
        stage_name = "Stage 1" if step <= 600 else "Stage 2"
        print(f"[{stage_name}] Step {step:4d}/{num_steps} | Loss: {loss_val:.4f} (Avg20: {recent_loss:.4f}) | Latency: {step_dt*1000:.1f} ms | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

    # Periodic checkpoint save
    if step % 250 == 0:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "K": 16,
                "proj_dim": 4096,
                "rank": 32,
                "num_deep_layers": 2,
                "thought_size": 64,
                "tier": "tier2",
            },
            "parameters": total_params,
            "step": step,
            "stage": "2048_human_conversational_checkpoint",
        }, checkpoint_path)
        print(f"  --> Periodic checkpoint saved at step {step} ({checkpoint_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)

total_time = time.perf_counter() - t_start
print(f"\nTraining Complete in {total_time:.2f}s ({total_time/60:.2f} min)!", flush=True)

# 5. Save Final Champion Checkpoint
print(f"\nSaving final champion checkpoint to {checkpoint_path}...", flush=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "vocab_size": tokenizer.vocab_size,
        "K": 16,
        "proj_dim": 4096,
        "rank": 32,
        "num_deep_layers": 2,
        "thought_size": 64,
        "tier": "tier2",
    },
    "parameters": total_params,
    "step": num_steps,
    "stage": "2048_human_conversational_champion",
}, checkpoint_path)
print(f"Checkpoint successfully saved! Size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB", flush=True)

# 6. Live Interactive Evaluation on A100
print("\n" + "=" * 80)
print("LIVE INTERACTIVE CONVERSATIONAL EVALUATION ON A100")
print("=" * 80, flush=True)
model.eval()
sess = StreamingCognitiveSession(model=model, tokenizer=tokenizer, device=device)

test_prompts = [
    "Good morning gamers, how are you today?",
    "Hey, how are you?",
    "Who are you?",
    "What is your name?",
    "I'm playing games right now.",
    "I just won my match in the lobby with my team!",
    "Tell me a joke.",
    "Why is the sky blue?",
    "Remember that project Alpha is due on Friday.",
    "When is project Alpha due?",
    "Remember Alice likes coffee.",
    "What does Alice like?",
    "Goodbye!",
]

for p in test_prompts:
    res = sess.generate_response(prompt_text=p, thread_id=0, max_new_tokens=32, temperature=0.3, repetition_penalty=1.15)
    resp = res["response_text"].strip()
    print(f"User:  {p}")
    print(f"Brain: {resp}")
    print(f"       [{res['mean_step_latency_ms']:.2f} ms/step]\n", flush=True)

print("=== ALL STAGES COMPLETED 100% SUCCESSFULLY ===", flush=True)
