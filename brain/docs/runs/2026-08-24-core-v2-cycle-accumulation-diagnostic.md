# Stage V2.0h cycle accumulation — rare pathways work, optimizer oscillates

**Date:** 2026-08-24  
**Release:** `r20260824t212624z-117918912777`  
**Archive SHA-256:** `1179189127770ca3993bf6ef261e30f5f3af09e32056255d5bc9dd9931129898`  
**Container:** `v20h-cycle-accumulation-20260824` / `08e78513b47c...`  
**Artifact:** `runs/v20h-cycle-accumulation-20260824/v20h_cycle_accumulation.json`  
**Status:** V2-A diagnostic only; not preregistered.

The exact macro gradient was averaged across all 47 batches, clipped once at
norm 1.0, and followed by one AdamW update per cycle. Sixteen cycles produced
752 batch exposures and 16 optimizer steps.

Macro probe loss improved from `1.87822` to a best `1.58096` at cycle 15 and
ended `1.60378`. Crucially, class 4 reached 100% training-bank recall at cycles
4–6 and class 0 reached 100% at cycles 8–10 and 15–16. Neither rare action had
ever been selected by the prior independently clipped runs. Thus accumulation
preserves the rare-class gradient pathway.

The solution oscillated between class-dominant states instead of learning all
classes simultaneously. Final training recall was `[1.0,.0466,0,.2162,0]`;
held-out recall was `[1.0,.019,0,.1709,0]`. Held-out balanced accuracy rose to
`0.2380`, but mean lift regressed to `-0.3233` because action 0 was emitted on
555/750 episodes.

The accumulated pre-clip norm fell from `789.57` at cycle 1 to mostly
`0.5–5.3` afterward. With AdamW lr `5e-4`, class preference continued to swing.
The next diagnostic compares lower learning rates under identical accumulated
gradients and longer 32-step exposure. No architecture or capability claim is
made from V2.0h.

