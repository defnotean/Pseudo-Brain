# Action-path overfit A/B record

Recorded: 2026-08-16

This experiment asked one narrow question: can the thesis-scale model learn
distinct W/A/S/D targets at all, or is its action path/objective broken? Both
runs used the same model seed, eight training sequences, 48 post-burn-in
decisions per optimizer step, 300 steps, a 10-step warmup, no weight decay, and
action-only gradients. The two immutable releases differed only in the
versioned action objective and its strict configuration surface.

## Immutable executions

| Run | Immutable release | Final checkpoint SHA-256 |
|---|---|---|
| A: legacy top-one hard negative | `r20260816t183929z-6242f62f95e4` | `70a94436587b3090120666597eecab9b0d574372f6c91259daaefaf28e154edb` |
| B: calibrated support plus smooth background tail | `r20260816t185304z-af37a86f160d` | `5f1907f11822e7f7bcbd9a3ea032c1591d30abbef98a35f4109fb604391b0da5` |

A archive SHA-256:
`6242f62f95e40fb919c8e8735c3f948002b7c6deed505f60516a75fcea63fafa`.
Its CUDA smoke checkpoint was
`18e4300d6896b105ac094b825684931ae8069cd98078e579afaf7b483e0e789f`
and its full-thesis canary checkpoint was
`6238f6f7738724e7ae51fae8e54635f2aa81a27d6022acd559b3e076e6b2c526`.

B archive SHA-256:
`af37a86f160d41c585d39be876242decd4967ed6f4aad6ccf1e6077d7e8571f1`.
Its CUDA smoke checkpoint was
`8e855700e453cc3b06bff1ebc83b4cdd473960279bd3f77bc147c07f3d4d3071`
and its full-thesis canary checkpoint was
`63410c624860bb6510c3e4f0cfa0383f04f3e24be2df0a26886eab26a4e1d840`.

## Result

Both action paths passed. At step 300 every anytime exit in both runs exactly
matched all four W/A/S/D bits and the full 296-button target on every one of the
48 repeated training decisions.

| Metric at train step 300 | A | B |
|---|---:|---:|
| Action loss | `0.00359675` | `0.00206485` |
| Exit 0/1/2/3 exact action set | `48/48` each | `48/48` each |
| Mean positive-key logit | `+6.1170` | `+6.0846` |
| Mean per-sample inactive movement maximum | `-6.2220` | `-6.2132` |
| Mean non-movement keyboard maximum | `-6.4336` | `-6.1549` |
| Opposite-direction conflicts | `0/48` | `0/48` |

A first crossed the precommitted `46/48` threshold between logged steps 160
and 180; B crossed it between steps 140 and 160. B therefore learned the tiny
set sooner while avoiding top-one winner switching and giving every configured
movement direction a gradient on every sample. It is selected for the broad
continuation gate.

This was deliberately not a generalization test. On eight disjoint validation
sequences, final-exit exactness at step 300 was `6/48` for A and `5/48` for B.
That low result prevents the overfit check from being misreported as a useful
game policy. The next gate trains on thousands of distinct sequences and uses a
fixed 96-decision held-out acceptance slice.

All accelerator work ran on the DGX Spark. The local PC performed no model
compute, screen/audio capture, or physical input injection. Both remote jobs
ended automatically and the Spark returned to idle.
