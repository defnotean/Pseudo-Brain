# Play-safe development policy

The owner may be playing a game while Pseudo-Brain is being developed. The
default development profile must not compete with the game for GPU time,
capture the display, inject input, install packages, or start persistent
services.

## Allowed without a separate performance window

- Editing source and documentation.
- Reading small repository files.
- CPU-only unit tests that finish in seconds.
- One CPU worker at below-normal process priority.
- Small deterministic simulations with no rendering window.

## Disabled while play-safe mode is active

- CUDA, DirectML, ROCm, or other accelerator use.
- Screen, audio, controller, or keyboard capture.
- Input injection.
- Training or model inference.
- Native compilation.
- Dependency installation or downloads.
- Video encoding.
- Profilers and sustained benchmarks.
- Background actors, services, or environment loops.

## Enforcement

Use scripts/run_play_safe_tests.ps1 for the Phase 0 test suite. It:

- Sets CUDA_VISIBLE_DEVICES to -1.
- Limits common numerical runtimes to one thread.
- Marks the process below normal priority.
- Runs only the deterministic unit-test directory.

The continuous driver also has a bounded catch-up ceiling. A large clock jump
halts and neutralizes the simulated controller instead of starting an
unbounded burst of work. Offline replay must opt in explicitly to unlimited
catch-up.

Performance work gets a separate configuration and must be started
deliberately when the owner is ready. No performance process should be launched
implicitly by importing the package.
