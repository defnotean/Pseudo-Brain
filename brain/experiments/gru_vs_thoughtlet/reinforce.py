"""REINFORCE-with-baseline fine-tuning for maze_chase BC policies.

Bolt-on policy gradient (no arch change — only needs logits, which forward gives):
- init from BC checkpoint, small LR, entropy bonus
- truncated BPTT: detach recurrent state every `seg_len` steps (same recipe as BC)
- short episodes (default 256 ticks) for Monte-Carlo returns without giant graphs
- rewards: pellet +1.0, caught -5.0, cleared +5.0, tick -0.002
- advantages standardized per batch; running return baseline logged
- families cycled round-robin; TRAIN seeds at --seed-offset (avoid dagger ranges)

Gate (a) target: 100 pel/1k + <20 catches/1k on registered families.
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BRAIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import make_model, count_parameters, N_FRAMES, ACTION_CLASSES
from train import get_device
from evaluate import _rgb_to_np
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.environments.pacman_harness import FAMILIES, partition_seeds, control_for_class


def rollout_episode(model, model_type, env, seed, device, n_ticks=256,
                    seg_len=32, temperature=1.0, ref_model=None):
    """On-policy rollout in seg_len-step segments (detach h between).

    Returns lists of per-step: logprob (tensor w/ grad), entropy (tensor),
    reward (float), kl-to-reference (tensor, 0 if no ref). h carried across
    segments (detached).
    """
    obs = env.reset(seed=seed)
    buf = [_rgb_to_np(obs.rgb)] * N_FRAMES
    prev_action = 0
    h = None
    logprobs, entropies, rewards, kls = [], [], [], []
    pellets = catches = 0

    for tick in range(n_ticks):
        frames_np = np.stack(buf)
        frames_t = torch.from_numpy(frames_np.transpose(0, 3, 1, 2)).unsqueeze(0).to(device)
        prev_t = torch.tensor([prev_action], dtype=torch.long, device=device)

        if tick % seg_len == 0 and h is not None:
            h = h.detach()

        if model_type == "reactive":
            logits, _ = model(frames_t, prev_t, None)
        elif model_type == "gru":
            if h is None and hasattr(model, "init_hidden"):
                h = model.init_hidden(1, device)
            logits, h = model(frames_t, prev_t, h)
        else:
            if h is None and hasattr(model, "init_thoughts"):
                h = model.init_thoughts(1, device)
            logits, h = model(frames_t, prev_t, h)

        probs = torch.softmax(logits / temperature, dim=-1).squeeze(0)
        dist_entropy = -(probs * torch.log(probs + 1e-8)).sum()
        action = torch.multinomial(probs, 1).item()
        logprob = torch.log(probs[action] + 1e-8)

        if ref_model is not None:
            # Approximate reference: same observation, zero recurrent state.
            # Carrying a parallel reference state doubles cost; zero-state KL
            # still anchors the policy and prevents destructive drift.
            with torch.no_grad():
                if model_type == "reactive":
                    ref_logits, _ = ref_model(frames_t, prev_t, None)
                elif model_type == "gru":
                    ref_logits, _ = ref_model(frames_t, prev_t, ref_model.init_hidden(1, device))
                else:
                    ref_logits, _ = ref_model(frames_t, prev_t, ref_model.init_thoughts(1, device))
            ref_logprobs = torch.log_softmax(ref_logits.squeeze(0), dim=-1)
            kl = (probs * (torch.log(probs + 1e-8) - ref_logprobs)).sum()
        else:
            kl = torch.tensor(0.0)

        step_out = env.step(control_for_class(action))
        r = -0.002
        if "pellet_eaten" in step_out.events:
            r += 1.0
            pellets += 1
        if "caught" in step_out.events:
            r += -5.0
            catches += 1
        if "cleared" in step_out.events:
            r += 5.0

        logprobs.append(logprob)
        entropies.append(dist_entropy)
        rewards.append(r)
        kls.append(kl)

        prev_action = action
        buf.append(_rgb_to_np(step_out.observation.rgb))
        buf = buf[-N_FRAMES:]
        if "cleared" in step_out.events:
            break
    return logprobs, entropies, rewards, kls, pellets, catches, tick + 1


def discounted_returns(rewards, gamma):
    out = [0.0] * len(rewards)
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="thoughtlet")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--ckpt-name", default="pg-thoughtlet")
    parser.add_argument("--save-dir", default=str(BRAIN_ROOT / "runs" / "gru_vs_thoughtlet"))
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--eps-per-update", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=256)
    parser.add_argument("--seg-len", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed-offset", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--kl-coef", type=float, default=0.0,
                        help="KL penalty to frozen BC reference (stops destructive drift)")
    parser.add_argument("--kl-ref-checkpoint", default=None)
    parser.add_argument("--resume-update", type=int, default=0,
                        help="continue counting/seeding from this update (ckpt already loaded via --init-checkpoint)")
    args = parser.parse_args()

    try:
        import os
        n_cores = int(os.environ.get("TORCH_NUM_THREADS", os.cpu_count() or 8))
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(max(1, n_cores // 2))
    except Exception:
        pass

    dev = get_device(args.device)
    model = make_model(args.model_type).to(dev)
    ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  Init: {args.init_checkpoint} | Params: {count_parameters(model)['total']:,}")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    baseline = 0.0
    save_path = Path(args.save_dir)
    log = {"updates": []}
    start = time.time()

    ref_model = None
    if args.kl_coef > 0 and args.kl_ref_checkpoint:
        ref_model = make_model(args.model_type).to(dev)
        ref_ckpt = torch.load(args.kl_ref_checkpoint, map_location="cpu", weights_only=False)
        ref_model.load_state_dict(ref_ckpt["state_dict"])
        ref_model.eval()
        print(f"  KL reference: {args.kl_ref_checkpoint} (coef={args.kl_coef})")

    for upd in range(args.resume_update + 1, args.updates + 1):
        all_lp, all_en, all_adv, all_kl = [], [], [], []
        ep_pellets, ep_catches, ep_ticks, ep_ret = [], [], [], []
        for e in range(args.eps_per_update):
            fam_idx = (upd * args.eps_per_update + e) % len(FAMILIES)
            fam = FAMILIES[fam_idx]
            seed = partition_seeds(fam_idx, "TRAIN")[(args.seed_offset - 500 + upd + e) % 524]
            env = MazeChaseEnv(**fam.env_kwargs())
            lp, en, rew, kl, pellets, catches, ticks = rollout_episode(
                model, args.model_type, env, seed, dev,
                n_ticks=args.ticks, seg_len=args.seg_len, temperature=args.temperature,
                ref_model=ref_model)
            rets = discounted_returns(rew, args.gamma)
            ep_ret.append(sum(rets) / max(len(rets), 1))
            all_lp.extend(lp)
            all_en.extend(en)
            all_kl.extend(kl)
            for i, g in enumerate(rets):
                all_adv.append(g)
            ep_pellets.append(pellets)
            ep_catches.append(catches)
            ep_ticks.append(ticks)

        adv = torch.tensor(all_adv, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        lp_t = torch.stack(all_lp)
        en_t = torch.stack(all_en)
        kl_t = torch.stack(all_kl).to(lp_t.device)
        # all_lp order matches all_adv order
        loss = (-(lp_t * adv.to(lp_t.device)).mean()
                - args.entropy_coef * en_t.mean()
                + args.kl_coef * kl_t.mean())

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        mean_ret = float(np.mean(ep_ret))
        baseline = 0.9 * baseline + 0.1 * mean_ret
        if upd % args.log_every == 0 or upd == 1:
            el = time.time() - start
            print(f"  upd {upd:4d}/{args.updates} loss={loss.item():+.4f} "
                  f"ret={mean_ret:+.3f} base={baseline:+.3f} "
                  f"pel={np.mean(ep_pellets):.1f} cat={np.mean(ep_catches):.2f} "
                  f"ent={en_t.mean().item():.3f} kl={kl_t.mean().item():.4f} {el:.0f}s", flush=True)
            log["updates"].append({"upd": upd, "loss": loss.item(), "ret": mean_ret,
                                   "pellets": float(np.mean(ep_pellets)),
                                   "catches": float(np.mean(ep_catches))})
        if upd % args.save_every == 0 or upd == args.updates:
            torch.save({"model_type": args.model_type, "state_dict": model.state_dict()},
                       save_path / f"{args.ckpt_name}.pt")
            with open(save_path / f"{args.ckpt_name}_pglog.json", "w") as f:
                json.dump(log, f, indent=2)

    print(f"DONE {time.time()-start:.0f}s -> {args.ckpt_name}.pt")


if __name__ == "__main__":
    main()
