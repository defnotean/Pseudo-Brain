# Remote Python execution for the independent pilot

Use only the authorized Colab/VPS Linux runtime. The implementation in
`evaluation/code_sandbox.py` fails closed without Bubblewrap, libseccomp and
the expected system Python3.12 runtime. There is no plain subprocess fallback.

The sandbox uses separate user, mount, PID, network, IPC and UTS namespaces,
no capabilities, a new session and parent-death cleanup. It exposes read-only
system Python/standard-library/runtime-library mounts, no host home/workspace,
no /proc, and no GPU devices. The sandbox root and /dev are read-only; /tmp has
a16 MiB ceiling. Environment is cleared. A syscall allowlist blocks socket,
fork/clone, namespace creation, ptrace and other unlisted operations. Python
runs with isolated mode, site initialization disabled, and bytecode writes off.
Limits:512 MiB address space,2 CPU seconds,6 wall seconds,64 KiB output/file size,
64 descriptors, and2 MiB combined program/input. Child creation is disallowed.

Remote environment: Ubuntu Bubblewrap0.9.0-1ubuntu0.1, system Python3.12.3.
User/network namespace probe succeeded. Colab disallows mounting another /proc
and changing user-namespace sysctls, so neither is used. The syscall filter
denies creation of further namespaces. Initial filters failed: Bubblewrap's
supervisor needed wait4; Python site startup attempted an NSS socket. wait4 was
allowed; site startup was disabled, preserving the network restriction.

Four integration tests passed in2.48s on Colab CPU. Frozen HumanEval controls:
all32 official canonical solutions passed and all32 return-None controls failed.
Normal grading, premature exit, filesystem/environment/device isolation,
network/process/namespace attempts, CPU, memory and output bounds were checked.
These are executor/grader controls, not trained-model scores.
Final module SHA256:
f6cdc00c7fc8133eea3d38381af7938ef34149e015b39cdfe1b109f4c083fb97.
Evidence/source (including failures):
`brain/runs/broad-pilot-20260911-v1/code-sandbox`; final evidence under `final/`.

The official check function executes in the same interpreter as the candidate.
This host-isolation setup is not an adversarially tamper-proof scoring service:
code intentionally inspecting/modifying the interpreter could compromise the
grader. Do not claim adversarial grading integrity. The benchmark inputs used
by model generation remain separate from all reference solutions and tests.

Primary documentation: https://github.com/containers/bubblewrap and
https://github.com/containers/bubblewrap/blob/main/bwrap.xml.
