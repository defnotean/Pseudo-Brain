# Decision Record: Local CPU as Canonical Compute; DGX Spark Reserved for Owner-Qwen Hosting

**Date**: August 26, 2026  
**Status**: APPROVED & REGISTERED ✅ (supersedes 2026-08-19)  
**Type**: Formal Hardware Architecture Amendment  
**Supersedes**: `2026-08-19-dgx-spark-primary-compute.md` (designating NVIDIA DGX Spark as primary compute)  
**Owner**: Pseudo-Brain autonomous construction run (owner contract, issued 2026-08-26)

---

## 1. Context

The 2026-08-19 record designated the NVIDIA DGX Spark as the unified compute
platform for all Pseudo-Brain training, inference, and actuation. That
designation is now obsolete on two independent grounds:

1. **Ownership conflict (binding).** The DGX Spark has been repurposed by the
   owner to host Qwen for the owner's own use. It is production hardware for a
   third party's workload and is **not** available for Pseudo-Brain compute.
   The owner's standing contract (2026-08-26) states this as non-negotiable:
   "NEVER launch anything on the DGX Spark — it is hosting Qwen for the owner."
2. **Empirical sufficiency [MEASURED].** The local Windows box (Ryzen 7
   9800X3D, 8C/16T, 31 GB RAM, torch 2.13.0+cpu, no CUDA) already carries the
   full program at scale: Core V1/V2 inference 2.27–3.00 ms per tick; PB21M's
   full 18-head × 73,728-step optimizer budget completed locally; post-hoc
   audits ran 126 s wall; the 44,161-parameter calibration pass and all
   probe-level retraining runs are local. Model-class budgets (~1M-parameter
   class) fit this host with multi-hour margins. No registered experiment in
   the PB21* series required Spark-class compute; several explicitly recorded
   `cuda_available: false` as an invariant.

## 2. Transition Summary

| Attribute | Previous Registered Target | New Registered Target |
| :--- | :--- | :--- |
| **Primary Compute Platform** | NVIDIA DGX Spark Unified Compute Platform | **Local CPU host (this Windows box)** |
| **Scope of Compute** | All training/inference/actuation on Spark | **All training, inference, audits, probes, qualification battery on local CPU** |
| **DGX Spark Role** | Pseudo-Brain compute | **Reserved: owner Qwen hosting. Pseudo-Brain must not schedule, launch, or inherit anything onto it.** |
| **Determinism** | (as before) | torch.use_deterministic_algorithms(True); CUBLAS_WORKSPACE_CONFIG=:4096:8; cudnn.deterministic=True, benchmark=False; tf32 off both paths; zlib.crc32(name) seed banks, never builtin hash() |

## 3. Invariants (binding)

- **I1 — Local canonical.** Every probe, trial, training run, and qualification
  gate executes on the local CPU host. Single-thread where determinism needs it.
- **I2 — Spark embargo.** The DGX Spark is read-only-inaccessible for compute:
  no SSH, no scheduler submission, no job inheritance, no "temporary" launch.
  Any code path that attempts to dispatch to it is a defect. This prevents a
  future agent from time-travelling back to the 2026-08-19 designation
  (constitution C14: compute-designation records are authoritative only for
  the date they carry; newest supersedes).
- **I3 — No silent platform drift.** Run provenance (run_provenance.py) records
  the platform string; any future amendment must be a new decision record in
  this directory, never an inline doc edit.

## 4. Consequences

- CURRENT_WORK.md and MASTER_ROADMAP hardware statements referencing
  "DGX Spark compute" are stale prose, superseded by this record; they remain
  verbatim as historical evidence.
- The two environmental test failures in `test_dgx_launch_contract.py` (POSIX
  bash contracts that only resolve on a Linux host) are **expected** failures
  on this Windows CPU host and are not regressions. They are documented here so
  a future agent does not "fix" them by re-adding Linux-only paths.
- The realtime-embodied-qualification v3 timing budgets (Q1 native-60Hz gates)
  are assessed against local CPU latency only; the 2026-08-19 Spark latency
  profile is withdrawn.

## 5. Verification

- Decision directory: `brain/docs/decisions/`
- Superseded record: `brain/docs/decisions/2026-08-19-dgx-spark-primary-compute.md`
- Owner contract anchor: owner instruction issued 2026-08-26, §0.
- Constitution anchor: ARCHITECTURAL_CONSTITUTION.md C14 (compute designations).
