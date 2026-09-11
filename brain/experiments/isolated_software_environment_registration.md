# Isolated software environment preparation

Implement a separate six-actuator environment with a virtual text workspace.
WRITE_FILE and exact EDIT_FILE alter inert bounded strings, READ_FILE returns
them, RETRIEVE_MEMORY calls an explicit provider, and FINISH requires an external
validator. A self-reported finish and a public-test pass cannot establish task
completion. No writes affect real project files or historical artifacts.

RUN_TESTS materializes a bounded snapshot for a read-only Bubblewrap mount.
Reuse the existing isolated Python3.12 runtime, namespace/seccomp controls and
512MiB memory/2s CPU/6s wall/64KiB output limits. No fallback to host execution.
The initial backend supports zero-argument test functions and unittest.TestCase;
fixtures, plugins and third-party dependencies are unavailable. Reject zero
discovered tests, skipped tests, expected failures and premature process exits.
Do not label this limited backend complete support for general software projects.
The same-interpreter harness remains explicitly not adversarially tamper-proof.

Workspace limits:64 files,64KiB/file,1MiB total, normalized relative POSIX paths,
no file/directory conflicts. Edits require exactly one target occurrence.
Invalid actions must leave files unchanged. External validators receive a copy;
their implementation must isolate any candidate execution. Retrieval and
validation providers are trusted harness components, not generated programs.

Run all checks remotely on Colab CPU, CUDA hidden, one thread: multi-file imports,
function/unittest discovery, wrong result then exact repair then independent
finish validation, unknown/malformed actions, path traversal and limits, no-test
and early-exit rejection, read-only snapshot/no host paths and blocked network.
These are scripted infrastructure controls, not learned autonomous capability.
Keep the active V5 GPU training and its source bundle unchanged.

First check outcome:2 positive controls failed,18 other controls passed. Both
positive executions exited159 with no output. The bootstrap called os.chdir
after the restrictive seccomp filter, whose allowlist intentionally omits chdir.
Preserve the failed v1 evidence. R1 moves working-directory setup into Bubblewrap
before the filter is applied; no syscall is added and no isolation limit changes.
Add explicit text for empty sandbox failure output and concise pytest case IDs;
all20 control inputs and acceptance criteria remain unchanged. Validate separately.
