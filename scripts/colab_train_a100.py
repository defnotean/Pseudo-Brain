"""Remote Google Colab A100 Training Script for Pseudo-Brain Tier 2 Conversational Core.

Executed on NVIDIA A100-SXM4-40GB GPU.
Fetches real multi-turn conversations from HuggingFace (UltraChat + Alpaca),
transforms them into multi-threaded cognitive episodes, trains the 25M-35M parameter core,
and saves the checkpoint for download.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

print("=" * 80)
print("PSEUDO-BRAIN CLOUD TRAINING PIPELINE ON NVIDIA A100-SXM4-40GB")
print("=" * 80)

# 1. Clone or pull repo
REPO_DIR = Path("/content/Pseudo-Brain")
if not REPO_DIR.exists():
    print("[1/5] Cloning repository...")
    subprocess.run(
        ["git", "clone", "-b", "defnotean/pseudo-brain", "https://github.com/defnotean/Pseudo-Brain.git", str(REPO_DIR)],
        check=True,
    )
else:
    print("[1/5] Updating repository...")
    subprocess.run(["git", "fetch", "origin"], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "checkout", "defnotean/pseudo-brain"], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "pull", "origin", "defnotean/pseudo-brain"], cwd=REPO_DIR, check=True)

# 2. Add repo to sys.path
sys.path.insert(0, str(REPO_DIR / "brain" / "src"))
sys.path.insert(0, str(REPO_DIR / "brain" / "experiments"))

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

corpus_path = build_and_cache_hf_corpus(
    loader=loader,
    transformer=transformer,
    num_ultrachat=200,
    num_alpaca=200,
    num_episodes=200,
    output_path=REPO_DIR / "brain" / "data" / "transformed_hf_conversational_corpus.jsonl",
)

# 4. Train Tier 2 Model on NVIDIA A100
checkpoint_dest = Path("/content/tier2_conversational_champion.pt")
print(f"\n[4/5] Training Tier 2 Model (proj_dim=2048, batch_size=8, steps=500)...")
train_metrics = train_tier2_directml(
    corpus_path=corpus_path,
    num_steps=500,
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
print("\n=== VERIFYING GENERATION ON A100 ===")
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

test_prompts = [
    "Hello! What is your name?",
    "Good morning gamers, how are you today?",
    "Remember that project Alpha is due on Friday.",
    "When is project Alpha due?",
    "What can you do?",
]

for p in test_prompts:
    res = sess.generate_response(prompt_text=p, max_new_tokens=32, temperature=0.7, repetition_penalty=1.35)
    print(f"Prompt:   {p}")
    print(f"Response: {res['response_text'].strip()}")
    print(f"Latency:  {res['mean_step_latency_ms']:.2f} ms/step\n")

print("=== ALL REMOTE CLOUD A100 OPERATIONS SUCCESSFUL ===")
