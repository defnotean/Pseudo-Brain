# Pseudo-Brain workspace rules

- This directory is the canonical workspace for the multi-thought
  sensorimotor-model project.
- Do not edit the sibling `Irene` project unless the user explicitly asks for
  an Irene change.
- Keep the historical `irene_brain` Python namespace until a versioned
  migration preserves checkpoint and release verification.
- Local verification must remain CPU-only, CUDA-hidden, single-threaded, and
  free of screen capture, HID injection, network services, or background jobs.
- Accelerator training runs only through the explicit, bounded DGX workflow.
- Preserve all historical run artifacts and registered configuration hashes.
- Do not start a longer run when a frozen scientific gate fails; diagnose or
  create a newly preregistered experiment instead.
- After each implementation, update the operator-facing documentation so it
  matches the code. When the owner has asked for continuous GitHub updates,
  commit and push that documentation with the change.
- Do not commit unless the user asks, except when the owner has explicitly
  requested continuous pushes.
