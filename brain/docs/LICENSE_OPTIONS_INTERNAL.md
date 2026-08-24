# LICENSE_OPTIONS_INTERNAL — INTERNAL ONLY, NO DECISION ENCODED

**Status:** informational. No license file has been added to the repo.
Choosing/applying one requires explicit owner instruction.

## Options compared

| License | Copyleft | Patent grant | Network/AGPL trap | Good fit if… |
|---|---|---|---|---|
| **Apache-2.0** | No | Explicit | N/A | Want permissive + patent protection; standard for ML repos |
| **MIT** | No | Implicit only | N/A | Maximum simplicity; fine if patent risk considered negligible |
| **GPL-3.0** | Strong (distribution) | Yes | No (triggers on distribution) | Want all downstream derivatives free |
| **AGPL-3.0** | Strongest (**network use counts**) | Yes | **Yes — SaaS use must open-source** | Want to prevent closed hosted clones of the model |
| **Source-available / noncommercial** (e.g. PolyForm NC / BSL) | N/A | varies | varies | Keep control; time-delayed or noncommercial-only release |
| **Dual license** (e.g. AGPL + commercial) | — | — | — | Open community + paid commercial path simultaneously |
| **Core proprietary + open tooling/benchmark** | — | — | — | Publish Torture Suite + harness + V1 history; keep V2 mechanisms private |

## Pseudo-Brain–specific considerations

1. **The benchmark is the crown jewel candidate.** Torture Suite + prereg
   discipline + provenance protocol are the most immediately valuable public
   artifacts and carry the least competitive risk. A "benchmark-only" release
   is viable at any time.
2. **V2 learning mechanisms are the IP-sensitive part** until confirmed,
   causally tested, and published. AGPL would force any hosted clone open —
   strongest protection; Apache/MIT give none.
3. **Patent/disclosure risk flag:** public disclosure of novel training
   methods can constitute a bar to later patent claims in many jurisdictions.
   If commercial protection is ever contemplated, get professional IP advice
   BEFORE any public release. This repo's commit history already contains
   detailed method descriptions — treat them as potentially disclosive now.
4. **History hygiene interacts with licensing:** a curated fresh-history
   public repo can carry a different license than this private archive.
   Nothing forces one license across both.

## Recommendation shape (NOT a decision)

Most likely end-states given owner goals:
- If priority = community adoption & safety review → Apache-2.0 on curated release.
- If priority = preventing closed hosted clones → AGPL-3.0.
- If priority = optionality → keep core private, open the benchmark + V1 corpus under Apache-2.0, decide V2 later.

Decision deferred to owner. Nothing applied.
