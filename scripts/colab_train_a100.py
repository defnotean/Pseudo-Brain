"""Remote Google Colab A100 Training Script for Pseudo-Brain Tier 2 Conversational Core.

Executed on NVIDIA A100-SXM4-40GB GPU.
Trains the 25M-35M parameter Tier 2 Pseudo-Brain on real-world multi-turn conversational
streams (UltraChat + Alpaca) + natural human chit-chat and multi-threaded cognitive tasks.
"""

import importlib
import json
import os
import sys
import time
import zipfile
import shutil
from pathlib import Path

print("=" * 80)
print("PSEUDO-BRAIN CLOUD TRAINING PIPELINE ON NVIDIA A100-SXM4-40GB")
print("=" * 80)

# 1. Unpack source code
REPO_DIR = Path("/content/Pseudo-Brain")
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
REPO_DIR.mkdir(parents=True, exist_ok=True)
brain_dir = REPO_DIR / "brain"
brain_dir.mkdir(parents=True, exist_ok=True)

zip_path = Path("/content/brain_src.zip")
print(f"[1/5] Unpacking brain package ({zip_path.stat().st_size / 1024:.1f} KB)...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(brain_dir)
print("      Brain package unpacked successfully.")

# 2. Add repo to sys.path and invalidate importlib caches
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

print(f"[2/5] PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"      Accelerator: {torch.cuda.get_device_name(0)}")
    print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# 3. Load Datasets and Build Transformed Corpus
from irene_brain.semantic.tokenizer import SemanticTokenizer
from irene_brain.semantic.native_semantic_model import make_semantic_model
from semantic_benchmark.hf_dataset_loader import HFConversationalLoader
from semantic_benchmark.llm_data_transform import LLMDataTransformer
from semantic_benchmark.train_conversational_tier2 import build_and_cache_hf_corpus, train_tier2_directml

print("\n[3/5] Building transformed multi-threaded cognitive corpus from HuggingFace...")
loader = HFConversationalLoader(cache_dir=REPO_DIR / "brain" / "data" / "huggingface")
transformer = LLMDataTransformer()

corpus_target = REPO_DIR / "brain" / "data" / "transformed_hf_conversational_corpus.jsonl"
if corpus_target.exists():
    corpus_target.unlink()

corpus_path = build_and_cache_hf_corpus(
    loader=loader,
    transformer=transformer,
    num_ultrachat=200,
    num_alpaca=200,
    num_episodes=200,
    output_path=corpus_target,
)

# Append diverse natural conversational templates to corpus
natural_dialogues = [
    ("Good morning gamers, how are you today?", "Good morning! I am doing great and ready to chat. How are you doing today?"),
    ("Hey, how are you?", "Hey! I am doing well, thank you. What are you up to?"),
    ("Hello! What is your name?", "Hello! I am Pseudo-Brain, an autonomous recurrent cognitive agent."),
    ("Who are you?", "I am Pseudo-Brain, a recurrent cognitive core running entirely on state memory without token replay buffers."),
    ("What are you doing right now?", "I am processing conversational thoughts and ready to assist you with anything you need."),
    ("I'm playing games right now.", "Nice! Enjoy your gaming session. What game are you playing?"),
    ("Can you hold a conversation like a normal person?", "Yes, absolutely! We can talk about games, science, projects, or anything on your mind."),
    ("What is your favorite topic?", "I enjoy discussing neuroscience, cognitive architectures, gaming, and creative problem solving."),
    ("Tell me something interesting.", "The human brain operates on approximately 20 watts of power while continuously running thousands of concurrent cognitive processes."),
    ("How does photosynthesis work?", "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll pigments."),
    ("Remember that project Alpha is due on Friday.", "I have recorded that project Alpha is due on Friday."),
    ("When is project Alpha due?", "Project Alpha is due on Friday."),
    ("Remember Alice likes coffee.", "Noted: Alice prefers coffee."),
    ("What does Alice like?", "Alice likes coffee."),
    ("Remember Bob likes tea.", "Noted: Bob prefers tea."),
    ("What does Bob like?", "Bob likes tea."),
    ("Let's plan a trip to Tokyo.", "Awesome plan! We can start by booking flights, picking hotels, and exploring iconic neighborhoods like Shibuya and Akihabara."),
    ("What can you do?", "I can reason across concurrent cognitive threads, retain long-term facts, and plan lookahead paths in real time."),
    ("Goodbye for now!", "Goodbye! Have fun playing your games, and let me know whenever you want to talk again."),
]

extra_items = []
for idx, (q, ans) in enumerate(natural_dialogues):
    for tid in range(2):
        turn_str = f"[THREAD:{tid}]{q} [RESP]{ans}[EOS]"
        toks = transformer.tokenizer.encode(turn_str)
        resp_id = transformer.tokenizer.resp_id
        targets = [-100] * len(toks)
        if resp_id in toks:
            r_idx = toks.index(resp_id)
            for ti in range(r_idx, len(toks) - 1):
                targets[ti] = toks[ti + 1]
        extra_items.append({
            "episode_id": f"natural_dialogue_{idx}_t{tid}",
            "text": turn_str,
            "tokens": toks,
            "targets": targets,
            "threads": [tid] * len(toks),
            "num_threads": 2,
            "has_preemption": False,
            "has_dependency": True,
        })

with open(corpus_path, "a", encoding="utf-8") as f:
    for item in extra_items:
        f.write(json.dumps(item) + "\n")
print(f"      Injected {len(extra_items)} natural conversational grounding episodes.")

# 4. Train Tier 2 Model on NVIDIA A100
checkpoint_dest = Path("/content/tier2_conversational_champion.pt")
print(f"\n[4/5] Training Tier 2 Model (proj_dim=2048, batch_size=8, steps=600)...")
train_metrics = train_tier2_directml(
    corpus_path=corpus_path,
    num_steps=600,
    batch_size=8,
    lr=3e-4,
    device_override="cuda:0",
    checkpoint_path=checkpoint_dest,
    proj_dim=2048,
    rank=32,
    num_deep_layers=2,
    threads=16,
)

print(f"\n[5/5] Training Complete!")
print(f"      Initial Loss: {train_metrics['initial_loss']:.4f}")
print(f"      Final Loss:   {train_metrics['final_loss']:.4f}")
print(f"      Total Time:   {train_metrics['total_time_s']:.2f} s")
print(f"      Throughput:   {train_metrics['throughput_tok_sec']:.1f} tok/s")
print(f"      Checkpoint:   {checkpoint_dest} ({checkpoint_dest.stat().st_size / (1024*1024):.2f} MB)")

# 5. Quick In-Colab Generation Verification
print("\n=== VERIFYING FLUENCY GENERATION ON A100 ===")
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
tok = SemanticTokenizer(max_threads=16)
ckpt = torch.load(checkpoint_dest, map_location=device, weights_only=False)
eval_model = make_semantic_model(
    model_type="pseudo_brain_tier2",
    vocab_size=tok.vocab_size,
    K=16,
    proj_dim=2048,
    rank=32,
    num_deep_layers=2,
    conditional_recurrence=True,
).to(device)
eval_model.load_state_dict(ckpt["model_state_dict"])
eval_model.eval()

sess = StreamingCognitiveSession(model=eval_model, tokenizer=tok, device=device)

eval_prompts = [
    "Good morning gamers, how are you today?",
    "Hey, how are you?",
    "Hello! What is your name?",
    "I'm playing games right now.",
    "Remember that project Alpha is due on Friday.",
    "When is project Alpha due?",
    "What can you do?",
]

for p in eval_prompts:
    res = sess.generate_response(prompt_text=p, thread_id=0, max_new_tokens=32, temperature=0.6, repetition_penalty=1.2)
    print(f"Prompt:   {p}")
    print(f"Response: {res['response_text'].strip()}")
    print(f"Latency:  {res['mean_step_latency_ms']:.2f} ms/step\n")

print("=== ALL REMOTE CLOUD A100 OPERATIONS SUCCESSFUL ===")
