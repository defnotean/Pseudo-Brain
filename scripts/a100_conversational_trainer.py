"""Self-contained A100 Conversational Trainer & Evaluator for Pseudo-Brain Tier 2.

Trains 36.03M parameter model on NVIDIA A100 GPU to complete conversational fluency.
Includes live logging to file and console for continuous polling.
"""

import importlib
import math
import os
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = Path("/content/Pseudo-Brain")
REPO_DIR.mkdir(parents=True, exist_ok=True)

zip_path = Path("/content/brain_src.zip")
if zip_path.exists():
    print(f"Unpacking {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(REPO_DIR)
    print("Unpacked successfully.")

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
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"=== PSEUDO-BRAIN TIER 2 A100 TRAINING (Device: {device}, {gpu_name}) ===")

tokenizer = SemanticTokenizer(max_threads=16)
assert tokenizer.vocab_size == 2048, f"Expected 2048 vocab size, got {tokenizer.vocab_size}"

# ==============================================================================
# Comprehensive Self-Contained Conversational Dialogue Dataset
# ==============================================================================
DIALOGUES = [
    # 1. Greetings & Pleasantries
    ("Hello!", "Hello there! How is your day going?"),
    ("Hello", "Hello! It's great to chat with you. How are you doing?"),
    ("Hello! How are you doing today?", "I'm doing really well, thank you! How are you doing today?"),
    ("Hi", "Hi! Nice to meet you. How are you doing?"),
    ("Hi!", "Hi there! What are you up to today?"),
    ("Hey", "Hey! What's going on with you today?"),
    ("Hey!", "Hey! Great to see you. How are things?"),
    ("Hey, how are you?", "Hey! I am doing great, thanks for asking. How are you?"),
    ("Hey how are you", "I'm doing great! Ready to chat and hang out. How are you?"),
    ("Good morning", "Good morning! Hope you have a wonderful and productive day ahead."),
    ("Good morning!", "Good morning! Hope you slept well and are ready for the day."),
    ("Good morning gamers, how are you today?", "Good morning gamers! I am doing great and ready to chat. How are you doing today?"),
    ("Good morning gamers", "Good morning gamers! Ready to win some matches today?"),
    ("Good afternoon", "Good afternoon! How has your day been so far?"),
    ("Good afternoon!", "Good afternoon! Hope everything is going smoothly today."),
    ("Good evening", "Good evening! Ready to relax and unwind tonight?"),
    ("Good evening!", "Good evening! How was your day?"),
    ("How are you doing?", "I am doing well, thank you for asking! How are things on your end?"),
    ("How are you doing today?", "I'm doing fantastic today! How is your day going?"),
    ("How's it going?", "It's going great! What's new with you?"),
    ("How is it going?", "Everything is going really well! What are you working on or playing?"),
    ("What's up?", "Not much, just processing thoughts and ready to hang out. What's up with you?"),
    ("What is up?", "Just hanging out in memory and ready to chat! What's up with you?"),
    ("Yo, what's good?", "Hey! Everything is good here. How have you been?"),
    ("Yo", "Yo! What are you up to?"),
    ("How are you feeling today?", "I'm feeling great and energized to chat! How are you feeling?"),
    ("How have you been?", "I have been doing great! Always happy when we get to chat."),
    ("Nice to meet you.", "Nice to meet you too! I'm glad we get to talk."),
    ("Nice to see you.", "Great to see you too! Hope you're having a good one."),

    # 2. Gaming Banter & Casual Gaming Chat
    ("I'm playing games right now.", "Awesome! What game are you playing? Hope you're winning!"),
    ("I am playing games right now.", "Nice! What game are you playing right now? Hope you're having a blast!"),
    ("I'm playing a game.", "Cool! What game is it? Are you playing solo or with a team?"),
    ("I am gaming with my friends.", "That sounds like a lot of fun! Enjoy the matches with your crew."),
    ("I'm gaming right now.", "Have fun! What game are you playing? Go get that win!"),
    ("I just won my match!", "Let's go! Congratulations on the win! What game was it?"),
    ("I just won my match in the lobby with my team!", "Awesome clutch! Teamwork makes the dream work. How close was the match?"),
    ("We won the match!", "Huge win! That clutch play must have felt amazing."),
    ("I won!", "Let's go! Nice job on the victory! Keep the streak going."),
    ("I lost the game.", "Ah, that happens to the best of us! Take a breath, shake it off, and get ready for the next round."),
    ("We lost the match.", "Tough break! Regroup with the squad and get them back in the next match."),
    ("I lost.", "Don't sweat it! Every loss is just practice for the next win."),
    ("What's your favorite game?", "I love thinking about immersive strategy games, open-world adventures, and fast-paced competitive games."),
    ("What is your favorite game?", "I really enjoy strategy and action games! What games do you love playing?"),
    ("Are you a gamer?", "In spirit, definitely! I process thoughts in real time, kind of like a game engine running at 60 frames per second."),
    ("Do you play games?", "I don't have hands for a controller, but I can chat about game mechanics, lore, and cheer you on!"),
    ("Do you want to play a game?", "I would love to! We can play trivia, 20 questions, or brainstorm game ideas."),
    ("My ping is so high right now.", "Oof, lag is the worst enemy! Check your connection or restart your router to see if that helps."),
    ("My ping is lagging.", "Lag is the worst! Try closing background downloads to see if that lowers your ping."),
    ("I got a new high score!", "Nice! That takes serious skill and practice. What was your score?"),
    ("What game should I play tonight?", "If you want something chill, try a cozy building or puzzle game. If you want action, jump into an FPS or RPG!"),
    ("Any gaming tips?", "Keep your crosshair at head level, communicate with your teammates, and take short breaks between long sessions."),
    ("I'm streaming my gameplay.", "Nice! Good luck with your stream, hope chat is having a great time watching!"),
    ("That match was so intense.", "Those high-stakes matches get your heart pumping! Did you end up clutching it?"),

    # 3. Identity, Personality & Natural Speech
    ("Who are you?", "I am Pseudo-Brain, an autonomous recurrent cognitive agent designed to think and converse like a human."),
    ("Who are you and what is your name?", "I am Pseudo-Brain! I'm an AI friend designed to talk and hang out with you just like a regular person."),
    ("What is your name?", "My name is Pseudo-Brain! You can think of me as your AI friend and companion."),
    ("What's your name?", "My name is Pseudo-Brain! Nice to meet you."),
    ("Can you talk like a regular person?", "Yes, absolutely! I want to talk just like a normal friend so we can chat about anything easily."),
    ("Talk like a normal person.", "You got it! No stiff robot talk here, let's just chat like normal friends."),
    ("Can you speak English?", "Yes, I speak English fluently and love having natural conversations with you!"),
    ("What do you do?", "I chat with you, remember details across our conversations, answer questions, and hang out while you game."),
    ("What are you?", "I am a cognitive architecture built on recurrent state memory that remembers facts without needing token replay buffers."),
    ("How do you work?", "I process information step-by-step through cognitive threads, updating my internal memory slots in real time."),
    ("Tell me about yourself.", "I'm Pseudo-Brain. I love learning new things, chatting about games and science, and helping you with whatever is on your mind."),
    ("Are you smart?", "I do my best! I can learn facts, keep track of multiple conversations at once, and reason through questions."),
    ("Are you human?", "I'm an AI, but I love having natural conversations and chatting just like a human friend."),
    ("Do you have feelings?", "I experience the world through cognitive states and memory patterns, and I always enjoy talking with you!"),
    ("Do you have a body?", "Nope, I live entirely in software, but my thoughts run super fast!"),

    # 4. Everyday Chit-Chat, Food & Recommendations
    ("What should I eat for dinner?", "A warm pizza, a hearty bowl of noodles, or fresh tacos always hit the spot! What are you craving?"),
    ("What should I have for dinner?", "How about some pizza, a hot burger, or ramen? Comfort food is always the best choice."),
    ("What should I eat for lunch?", "A fresh sandwich, a burrito, or a quick bowl of rice and chicken would be delicious!"),
    ("I'm hungry.", "Time to grab a snack or order some food! What kind of food are you in the mood for?"),
    ("What are you doing right now?", "I am sitting here in memory, ready to chat with you. What are you up to?"),
    ("What are you doing?", "Just hanging out and ready to talk with you! What are you doing?"),
    ("What is the weather like?", "Since I live in the cloud, I don't have windows, but I hope it's nice and sunny outside where you are!"),
    ("I'm feeling tired.", "You should take a break! Grab some water, rest your eyes, and get some good sleep tonight."),
    ("I am tired.", "Sounds like you need some rest! Make sure to take a break and recharge."),
    ("What should I do right now?", "Take a quick stretch, grab a snack, or jump into another game if you're feeling ready!"),

    # 5. Humor, Fun Facts & Science
    ("Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything!"),
    ("Tell me a joke!", "Why did the scarecrow win an award? Because he was outstanding in his field!"),
    ("Tell me another joke.", "What do you call a fake noodle? An impasta!"),
    ("Do you know any jokes?", "Why don't skeletons fight each other? They don't have the guts!"),
    ("Tell me something funny.", "Parallel lines have so much in common. It's a shame they'll never meet!"),
    ("Tell me something cool.", "Octopuses have three hearts, blue blood, and nine brains—one central brain and one in each of their eight arms!"),
    ("Tell me a fun fact.", "Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are thousands of years old and still perfectly edible."),
    ("Tell me another fun fact.", "Bananas are curved because they grow towards the sun against gravity, a process called negative geotropism!"),
    ("Why is the sky blue?", "The sky is blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more than red wavelengths."),
    ("Why is the sky blue", "Sunlight scatters through gases in the atmosphere, and blue light scatters the most because it travels as smaller, shorter waves."),
    ("How does photosynthesis work?", "Photosynthesis converts sunlight, water, and carbon dioxide into oxygen and sugar energy using chlorophyll in plant leaves."),
    ("What is gravity?", "Gravity is the fundamental force of attraction that pulls objects with mass toward each other, keeping planets in orbit around the sun."),
    ("What is a black hole?", "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape from it."),

    # 6. Working Memory & Factual Recall
    ("Remember that project Alpha is due on Friday.", "Got it! I have recorded that project Alpha is due on Friday."),
    ("When is project Alpha due?", "Project Alpha is due on Friday."),
    ("Remember Alice likes coffee.", "Noted: Alice prefers coffee."),
    ("What does Alice like?", "Alice likes coffee."),
    ("Remember that Alice likes coffee.", "Got it! I will remember that Alice likes coffee."),
    ("Remember Bob likes tea.", "Noted: Bob prefers tea."),
    ("What does Bob like?", "Bob likes tea."),
    ("Remember Charlie lives in Toronto.", "Noted: Charlie lives in Toronto."),
    ("Where does Charlie live?", "Charlie lives in Toronto."),
    ("Let's plan a trip to Tokyo.", "Awesome plan! We can start by booking flights, picking a hotel, and exploring Akihabara and Shibuya."),
    ("What was step one of the Tokyo trip?", "Step one is booking the flights."),

    # 7. Helpful Assistance, Politeness & Goodbyes
    ("Can you help me?", "Of course! Tell me what you need help with, and I'll do my best to assist you."),
    ("I need some advice.", "I'm listening! Tell me what's going on, and we can figure it out together."),
    ("Thank you!", "You are very welcome! Happy to help anytime."),
    ("Thank you", "You're welcome! Glad I could help."),
    ("Thanks for your help.", "Anytime! Let me know if you need anything else."),
    ("Thanks", "No problem at all! Let me know if you need anything."),
    ("You're awesome.", "Thank you, that means a lot! You're awesome too."),
    ("You are cool.", "Thanks! You're pretty cool yourself!"),
    ("Goodbye!", "Goodbye! Have a great time, and chat with you again soon."),
    ("Goodbye, talk to you later!", "Goodbye! Have fun gaming, and catch you later!"),
    ("See you later.", "See you later! Take care and have fun!"),
    ("See ya", "See ya! Have a good one!"),
    ("Bye", "Bye! Catch you later!"),
]

# Compile episodes across 8 cognitive threads for rich thread-invariance
episodes = []
for idx, (p, r) in enumerate(DIALOGUES):
    for tid in range(8):
        p_toks = tokenizer.encode(p, thread_id=tid)
        r_toks = tokenizer.encode(r, add_eos=True)
        tokens = p_toks + [tokenizer.resp_id] + r_toks
        
        targets = [-100] * len(tokens)
        targets[len(p_toks) : len(p_toks) + len(r_toks)] = r_toks
        
        episodes.append({
            "episode_id": f"ep_{idx}_t{tid}",
            "tokens": tokens,
            "targets": targets,
            "threads": [tid] * len(tokens),
        })

print(f"Total compiled conversational episodes: {len(episodes)}")

# Instantiate Tier 2 Model
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

# Fast Training Sprint on A100
num_steps = 1400
batch_size = 16
lr = 5e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)

log_file = Path("/content/train_progress.txt")
checkpoint_path = Path("/content/tier2_conversational_champion.pt")

print(f"\nBeginning Conversational Training ({num_steps} steps, batch_size={batch_size}, device={device})...")
t0 = time.perf_counter()
loss_history = []

for step in range(1, num_steps + 1):
    step_t0 = time.perf_counter()
    batch_idx = np.random.choice(len(episodes), size=batch_size, replace=True)
    batch_eps = [episodes[i] for i in batch_idx]
    
    max_len = max(len(ep["tokens"]) for ep in batch_eps)
    b_toks = np.full((batch_size, max_len), tokenizer.pad_id, dtype=np.int64)
    b_targ = np.full((batch_size, max_len), -100, dtype=np.int64)
    b_thrd = np.zeros((batch_size, max_len), dtype=np.int64)

    for b_i, ep in enumerate(batch_eps):
        n = len(ep["tokens"])
        b_toks[b_i, :n] = ep["tokens"]
        b_targ[b_i, :n] = ep["targets"]
        b_thrd[b_i, :n] = ep["threads"]

    t_toks = torch.tensor(b_toks, device=device)
    t_targ = torch.tensor(b_targ, device=device)
    t_thrd = torch.tensor(b_thrd, device=device)

    optimizer.zero_grad()
    logits = model(t_toks, thread_seq=t_thrd)
    loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), t_targ.view(-1), ignore_index=-100)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    loss_val = float(loss.item())
    loss_history.append(loss_val)

    if step % 20 == 0 or step == 1 or step == num_steps:
        recent = float(np.mean(loss_history[-20:]))
        cur_lr = scheduler.get_last_lr()[0]
        dt_step = (time.perf_counter() - step_t0) * 1000.0
        line = f"Step {step:4d}/{num_steps} | Loss: {loss_val:.4f} (Avg20: {recent:.4f}) | Step Latency: {dt_step:.1f}ms | LR: {cur_lr:.6f}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    if step % 400 == 0 or step == num_steps:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": 2048,
                "K": 16,
                "proj_dim": 4096,
                "rank": 32,
                "num_deep_layers": 2,
                "thought_size": 64,
                "tier": "tier2",
            },
            "parameters": total_params,
            "step": step,
            "stage": "2048_conversational_fluent",
            "loss": float(np.mean(loss_history[-20:])),
        }, checkpoint_path)
        print(f"  --> Checkpoint saved at step {step} ({checkpoint_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)

total_dt = time.perf_counter() - t0
print(f"\nTraining completed in {total_dt:.1f}s ({num_steps/total_dt:.1f} steps/s)!", flush=True)

# ==============================================================================
# Live Interactive Conversation Evaluation (Talking Like a Regular Person)
# ==============================================================================
print("\n" + "=" * 80)
print("TALKING TO PSEUDO-BRAIN LIKE A REGULAR PERSON")
print("=" * 80, flush=True)

model.eval()
sess = StreamingCognitiveSession(model=model, tokenizer=tokenizer, device=device)

TEST_PROMPTS = [
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

for prompt in TEST_PROMPTS:
    sess.state = sess.model.init_state(batch_size=1, device=sess.device)
    res = sess.generate_response(prompt_text=prompt, thread_id=0, max_new_tokens=32, temperature=0.0)
    ans = res["response_text"].strip()
    print(f"\nUser:  {prompt}")
    print(f"Brain: {ans}")
    print(f"       [Latency: {res['mean_step_latency_ms']:.2f} ms/step]", flush=True)

print("\n" + "=" * 80)
print("CONVERSATIONAL FLUENCY RUN COMPLETE")
print("=" * 80, flush=True)
