"""Proper PPO for maze_chase BC policies (recurrent, full-episode).

Why this exists: REINFORCE (short 256-tick eps, MC returns, no critic)
destroyed good BC inits. This implements the standard recurrent-PPO recipe:
- FULL-EPISODE rollouts (to clear/cap, per-family max_ticks) on TRAIN seeds
- learned value baseline (ThoughtletModel.value_head, zero-init)
- GAE advantages per episode (gamma/lambda), bootstrap 0 (episodes terminate)
- truncated-BPTT minibatches: segments of SEG steps with stored h_in,
  h_in detached (no grad across segments) — exact on-policy recomputation
- clipped surrogate + clipped value + entropy bonus
- EXACT KL-to-BC penalty: frozen ref carries its own recurrent state in
  parallel during rollout; per-step ref logprobs stored, KL(new||ref) exact
- deterministic TRAIN-seed schedule; eval on DEV partition (held-out layouts)

Init: best BC checkpoint (long15k-thoughtlet-142.pt), strict=False load
(value_head keys absent in BC ckpt).
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


def shaped_reward(events) -> tuple[float, int, int, int]:
    r = -0.002
    pel = 1 if "pellet_eaten" in events else 0
    cat = 1 if "caught" in events else 0
    clr = 1 if "cleared" in events else 0
    r += pel * 1.0 - cat * 5.0 + clr * 5.0
    return r, pel, cat, clr


@torch.no_grad()
def rollout_episode(model, ref_model, env, seed, device, temperature=1.0, max_ticks=10000):
    """Full-episode on-policy rollout.

    Returns dict of per-step lists: obs (uint8 [N_FRAMES,3,16,16]),
    prev_action, action, logprob, value, ref_logprob[5], reward;
    plus h_inits: {seg_start_tick: h tensor} for truncated-BPTT updates,
    totals (pellets, catches, ticks).
    """
    obs = env.reset(seed=seed)
    buf = [_rgb_to_np(obs.rgb)] * N_FRAMES
    prev_action = 0
    h = h_ref = None
    steps = []
    h_inits = {}
    pellets = catches = 0
    nan_ticks = 0
    tick = 0
    model.eval()

    while True:
        frames_np = np.stack(buf)  # (4,16,16,3) float [0,1]
        frames_t = torch.from_numpy(frames_np.transpose(0, 3, 1, 2)).unsqueeze(0).float().to(device)
        prev_t = torch.tensor([prev_action], dtype=torch.long, device=device)

        if tick % SEG_LEN == 0:
            h_inits[tick] = h.detach().clone() if h is not None else None

        logits, h_next, value = model.actor_critic(frames_t, prev_t, h)
        logits = logits.squeeze(0)
        if not torch.isfinite(logits).all():
            nan_ticks += 1
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        probs = torch.softmax(logits / temperature, dim=-1)
        s = probs.sum()
        probs = probs / s if torch.isfinite(s) and s.item() > 0 else torch.ones_like(probs) / probs.numel()
        action = torch.multinomial(probs, 1).item()
        logprob = torch.log(probs[action] + 1e-8).item()

        with torch.no_grad():
            ref_logits, h_ref = ref_model(frames_t, prev_t, h_ref)
            ref_logprobs = torch.log_softmax(ref_logits.squeeze(0), dim=-1).cpu().numpy()

        step_out = env.step(control_for_class(action))
        r, pel, cat, clr = shaped_reward(step_out.events)
        pellets += pel
        catches += cat

        steps.append({
            "obs": (frames_np * 255).astype(np.uint8),
            "prev": prev_action, "action": action,
            "logprob": logprob, "value": value.item(),
            "ref_lp": ref_logprobs.astype(np.float32), "reward": r,
        })
        h = h_next.detach()
        h_ref = h_ref.detach()
        prev_action = action
        buf.append(_rgb_to_np(step_out.observation.rgb))
        buf = buf[-N_FRAMES:]
        tick += 1
        if clr or tick >= max_ticks:
            break
    h_stats = {}
    if h is not None:
        hf = h.detach().float().cpu().numpy()
        h_stats = {"h_mean": float(hf.mean()), "h_std": float(hf.std()),
                   "h_absmax": float(np.abs(hf).max())}
    return steps, h_inits, pellets, catches, tick, nan_ticks, h_stats


def compute_gae(rewards, values, gamma, lam):
    """Full-episode GAE, terminal bootstrap 0. Returns (advantages, returns)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
    return adv, adv + np.array(values, dtype=np.float32)


def build_segments(episodes, seg_len):
    """Split episodes into (ep_idx, start) segments with stored h_in."""
    segs = []
    for ei, (steps, h_inits, _, _, _) in enumerate(episodes):
        for s in range(0, len(steps), seg_len):
            segs.append((ei, s, h_inits[s]))
    return segs


def segment_batch(model, episodes, seg_list, device):
    """Batched truncated-BPTT recomputation (graph attached).

    Minibatch segments run as batch dim B= affiliates; recurrence steps t=0..T-1
    sequentially with carried h (exact on-policy recompute from stored h_in).
    Tail segments shorter than SEG_LEN are padded; mask excludes padding.
    Returns (new_logp_sel, entropy, values, mask) flat over valid steps.
    """
    B = len(seg_list)
    K = model.K if hasattr(model, "K") else None
    segs = [episodes[ei][0][s:s + SEG_LEN] for ei, s, _ in seg_list]
    T = max(len(sg) for sg in segs)
    h = None
    if seg_list[0][2] is not None:
        h = torch.cat([h_in if h_in is not None else torch.zeros_like(seg_list[0][2])
                       for _, _, h_in in seg_list], dim=0).to(device)
    lp_list, en_list, v_list, m_list = [], [], [], []
    for t in range(T):
        ot, pv, m = [], [], []
        for sg in segs:
            if t < len(sg):
                ot.append(sg[t]["obs"])
                pv.append(sg[t]["prev"])
                m.append(1.0)
            else:
                ot.append(sg[-1]["obs"])
                pv.append(sg[-1]["prev"])
                m.append(0.0)
        obs = np.stack(ot).astype(np.float32) / 255.0
        obs_t = torch.from_numpy(obs.transpose(0, 1, 4, 2, 3)).float().to(device)
        prev_t = torch.tensor(pv, dtype=torch.long, device=device)
        logits, h, value = model.actor_critic(obs_t, prev_t, h)
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        logp_all = torch.log_softmax(logits, dim=-1)
        at = torch.tensor([sg[t]["action"] if t < len(sg) else 0 for sg in segs],
                          dtype=torch.long, device=device)
        sel = logp_all[torch.arange(B, device=device), at]
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * logp_all).sum(-1)
        mask = torch.tensor(m, dtype=torch.float32, device=device)
        lp_list.append(sel * mask)
        en_list.append(ent * mask)
        v_list.append(value * mask)
        m_list.append(mask)
    return (torch.cat(lp_list), torch.cat(en_list), torch.cat(v_list), torch.cat(m_list))


def seg_targets(episodes, seg_list, device):
    """Gather stored old_logprob / ref_logprob(sel) / adv / ret, padded like segment_batch."""
    segs = [episodes[ei][0][s:s + SEG_LEN] for ei, s, _ in seg_list]
    T = max(len(sg) for sg in segs)
    olp, rlp, adv, ret, m = [], [], [], [], []
    for sg in segs:
        for t in range(T):
            if t < len(sg):
                st = sg[t]
                olp.append(st["logprob"])
                rlp.append(float(st["ref_lp"][st["action"]]))
                adv.append(st["adv"])
                ret.append(st["ret"])
                m.append(1.0)
            else:
                olp.append(0.0)
                rlp.append(0.0)
                adv.append(0.0)
                ret.append(0.0)
                m.append(0.0)
    return (torch.tensor(olp, dtype=torch.float32, device=device),
            torch.tensor(rlp, dtype=torch.float32, device=device),
            torch.tensor(adv, dtype=torch.float32, device=device),
            torch.tensor(ret, dtype=torch.float32, device=device),
            torch.tensor(m, dtype=torch.float32, device=device))


SEG_LEN = 128  # module-level so rollout_episode can use it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="thoughtlet")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--ckpt-name", default="ppo-thoughtlet")
    parser.add_argument("--save-dir", default=str(BRAIN_ROOT / "runs" / "gru_vs_thoughtlet"))
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--eps-per-update", type=int, default=5)
    parser.add_argument("--seg-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-segs", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--families", nargs="+", type=int, default=None,
                        help="subset of family indices (smoke tests)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    global SEG_LEN
    SEG_LEN = args.seg_len

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
    missing = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"  Init: {args.init_checkpoint} | Params: {count_parameters(model)['total']:,}")
    print(f"  (value_head fresh: {list(missing.missing_keys)})")
    ref_model = make_model(args.model_type).to(dev)
    ref_model.load_state_dict(ckpt["state_dict"], strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    save_path = Path(args.save_dir)
    log = {"updates": [], "args": vars(args)}
    start = time.time()
    fam_ids = args.families if args.families else list(range(len(FAMILIES)))

    for upd in range(1, args.updates + 1):
        # ---- collect: full episodes, deterministic TRAIN schedule ----
        episodes = []
        ep_pel, ep_cat, ep_ticks = [], [], []
        ep_nan, ep_hmean, ep_hstd, ep_hmax = [], [], [], []
        ep_pel1k, ep_cat1k = [], []
        for e in range(args.eps_per_update):
            fam_idx = fam_ids[(upd * args.eps_per_update + e) % len(fam_ids)]
            fam = FAMILIES[fam_idx]
            train_seeds = partition_seeds(fam_idx, "TRAIN")
            seed = train_seeds[(args.seed_offset + upd * args.eps_per_update + e) % len(train_seeds)]
            env = MazeChaseEnv(**fam.env_kwargs())
            steps, h_inits, pel, cat, ticks, nan_t, h_stats = rollout_episode(
                model, ref_model, env, seed, dev, temperature=args.temperature,
                max_ticks=fam.max_ticks)
            rew = [st["reward"] for st in steps]
            val = [st["value"] for st in steps]
            adv, ret = compute_gae(rew, val, args.gamma, args.lam)
            for st, a, r in zip(steps, adv, ret):
                st["adv"] = float(a)
                st["ret"] = float(r)
            episodes.append((steps, h_inits, pel, cat, ticks))
            ep_pel.append(pel)
            ep_cat.append(cat)
            ep_ticks.append(ticks)
            ep_nan.append(nan_t)
            ep_pel1k.append(pel / max(ticks, 1) * 1000)
            ep_cat1k.append(cat / max(ticks, 1) * 1000)
            ep_hmean.append(h_stats.get("h_mean", 0.0))
            ep_hstd.append(h_stats.get("h_std", 0.0))
            ep_hmax.append(h_stats.get("h_absmax", 0.0))

        # explained variance of the critic on this batch (1 = perfect)
        all_ret = np.concatenate([np.array([st["ret"] for st in ep[0]]) for ep in episodes])
        all_val = np.concatenate([np.array([st["value"] for st in ep[0]]) for ep in episodes])
        ev = 1.0 - np.var(all_ret - all_val) / max(np.var(all_ret), 1e-8)

        # normalize advantages across the update batch
        all_adv = np.concatenate([np.array([st["adv"] for st in ep[0]]) for ep in episodes])
        am, ast = all_adv.mean(), all_adv.std() + 1e-8
        for ep in episodes:
            for st in ep[0]:
                st["adv"] = (st["adv"] - am) / ast

        segs = build_segments(episodes, args.seg_len)
        model.train()
        tot = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "approxkl": 0.0,
               "grad_norm": 0.0, "clipfrac": 0.0, "n": 0}
        for epoch in range(args.epochs):
            order = np.random.permutation(len(segs))
            for i in range(0, len(segs), args.minibatch_segs):
                mb = [segs[j] for j in order[i:i + args.minibatch_segs]]
                new_lp, ent, new_v, mask = segment_batch(model, episodes, mb, dev)
                old_lp, ref_lp, adv_t, ret_t, _ = seg_targets(episodes, mb, dev)
                msum = mask.sum().clamp_min(1.0)

                logr_bc = new_lp - ref_lp  # log pi_new(a) - log pi_ref(a)
                r_bc = torch.exp(logr_bc)
                # Schulman k3 estimator of KL(new||ref): E[r - 1 - log r] >= 0
                kl_bc_step = r_bc - 1.0 - logr_bc

                ratio = torch.exp(new_lp - old_lp)
                pg1 = ratio * adv_t
                pg2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t
                pg_loss = -(torch.min(pg1, pg2) * mask).sum() / msum
                v_clipped = ret_t + torch.clamp(new_v - ret_t, -args.clip, args.clip)
                vf_loss = (torch.max((new_v - ret_t) ** 2, (v_clipped - ret_t) ** 2) * mask).sum() / msum
                ent_b = (ent * mask).sum() / msum
                kl_bc = (kl_bc_step * mask).sum() / msum
                approxkl_old_new = ((ratio - 1.0) - torch.log(ratio.clamp_min(1e-8))).mean()
                loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * ent_b + args.kl_coef * kl_bc

                opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

                tot["pg"] += pg_loss.item()
                tot["vf"] += vf_loss.item()
                tot["ent"] += ent_b.item()
                tot["kl"] += kl_bc.item()
                tot["approxkl"] += approxkl_old_new.item()
                tot["grad_norm"] += float(grad_norm)
                tot["clipfrac"] += ((((ratio - 1).abs() > args.clip).float() * mask).sum() / msum).item()
                tot["n"] += 1

        if upd % args.log_every == 0 or upd == 1:
            n = tot["n"]
            el = time.time() - start
            print(f"  upd {upd:4d}/{args.updates} pg={tot['pg']/n:+.4f} vf={tot['vf']/n:.4f} "
                  f"ent={tot['ent']/n:.3f} kl={tot['kl']/n:.4f} akl={tot['approxkl']/n:.4f} "
                  f"gn={tot['grad_norm']/n:.2f} clip={tot['clipfrac']/n:.2f} ev={ev:+.3f} "
                  f"pel/1k={np.mean(ep_pel1k):.1f} cat/1k={np.mean(ep_cat1k):.1f} "
                  f"nan={int(np.sum(ep_nan))} h={np.mean(ep_hmean):+.3f}/{np.mean(ep_hstd):.3f}/{np.mean(ep_hmax):.2f} "
                  f"{el:.0f}s",
                  flush=True)
            log["updates"].append({"upd": upd, "pg": tot["pg"] / n, "vf": tot["vf"] / n,
                                   "kl": tot["kl"] / n, "approxkl": tot["approxkl"] / n,
                                   "grad_norm": tot["grad_norm"] / n,
                                   "clipfrac": tot["clipfrac"] / n, "explained_var": float(ev),
                                   "pellets_per_1k": float(np.mean(ep_pel1k)),
                                   "catches_per_1k": float(np.mean(ep_cat1k)),
                                   "nan_ticks": int(np.sum(ep_nan)),
                                   "h_mean": float(np.mean(ep_hmean)),
                                   "h_std": float(np.mean(ep_hstd)),
                                   "h_absmax": float(np.mean(ep_hmax)),
                                   "pellets": float(np.mean(ep_pel)),
                                   "catches": float(np.mean(ep_cat)),
                                   "ticks": float(np.mean(ep_ticks))})
        if upd % args.save_every == 0 or upd == args.updates:
            torch.save({"model_type": args.model_type, "state_dict": model.state_dict(),
                        "ppo_update": upd, "ppo_args": vars(args),
                        "init_checkpoint": args.init_checkpoint},
                       save_path / f"{args.ckpt_name}.pt")
            with open(save_path / f"{args.ckpt_name}_ppolog.json", "w") as f:
                json.dump(log, f, indent=2)

    print(f"DONE {time.time()-start:.0f}s -> {args.ckpt_name}.pt")


if __name__ == "__main__":
    main()
