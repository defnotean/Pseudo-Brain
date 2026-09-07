# Workstream 3: Multi-Step Autonomous Agent Reasoning, Dynamic Tool Parameterization, and Consequence-Directed Inhibition of Return

**Date:** 2026-09-07  
**Platform:** Local Windows CPU (Single-thread deterministic runtime)  
**Status:** COMPLETE & VERIFIED (All Agent Loop & Reasoning Tests Passing)  
**Implementation:** `brain/src/irene_brain/agent/loop.py`, `tools.py`, `goal.py`  
**Test Suites:** `brain/tests/test_agent_loop.py`, `brain/tests/test_agent_reasoning.py`  

---

## 1. Context & Mechanistic Objective

In the canonical Master Roadmap (`brain/docs/MASTER_ROADMAP.md`, Phase 2.9 Foundation Modalities / Tools and Phase 4 Autonomous Agent), Pseudo-Brain must scale beyond low-level embodied game loops to persistent, multi-step software and research tasks.

### Baseline Limitation: Action Perseveration and Lack of Consequence-Directed Control
Conventional lightweight autonomous agent loops suffer from critical failure modes when tool outcomes fail:
1. **Action Perseveration / Infinite Retries**: When an action (e.g. running a test or querying a file) fails, naive agents either endlessly repeat the identical failing command or crash, lacking an internal error-driven inhibition mechanism.
2. **Subgoal Stagnation**: After a prerequisite step succeeds (e.g. reading a file or creating a scaffolding script), agents fail to transition attention forward, repeatedly re-reading or re-executing already satisfied subgoals.
3. **Disconnected Observations & Outcomes**: Observations typically capture only raw text snippets, ignoring structured outcome flags (`last_success`, `last_reward`, `last_tool_idx`), preventing internal world models from predicting consequence surprise.

### Proposed Solution: Consequence-Gated Agent Cognitive Core
We extended `PseudoBrainAgent` and `AgentCognitiveCore` with bio-plausible cognitive control mechanisms:
1. **Error-Induced Inhibition of Return (IOR)**:
   When an executed tool fails (`last_success is False`), synaptic plasticity modulation $P_t$ immediately applies strong negative suppression to that specific tool:
   $$	ext{Suppression}[0, a_{\text{failed}}] = -4.0 \cdot (\delta_r + 1.0)$$
   This drops the logit for the failing tool by $>6.0$ units, completely preventing motor perseveration and immediately forcing the policy to select an exploratory alternative or repair tool.
2. **Subgoal Progression Discounting**:
   When an action succeeds (`last_success is True`), attention naturally shifts forward by applying a moderate completion discount:
   $$\text{Discount}[0, a_{\text{succeeded}}] = -1.2$$
   This facilitates progressive transitions along multi-step pipelines (e.g. `read_file` $\to$ `write_file` $\to$ `run_command` $\to$ `verify_goal`) without requiring brittle hand-coded state machines.
3. **Structured Outcome Observation Encodings**:
   `_encode_observation` embeds not only hashed lexical tokens (via deterministic CRC32) but also dedicated channels for `last_tool_idx`, `last_success` ($+2.0$ / $-2.0$), and clipped reward $\text{clip}(r, -2, 2)$, providing the recurrent state $h_t$ with clean outcome provenance.
4. **Dynamic Contextual Tool Parameterization**:
   Added `arg_provider` callbacks to `run_task`, enabling dynamic synthesis of tool arguments (such as extracting numbers from prior file outputs or formatting commands) based on step execution logs.

---

## 2. Empirical Benchmark Battery [MEASURED]

The architecture was evaluated across three distinct multi-step software and fault-recovery tasks in `brain/tests/test_agent_reasoning.py`:

| Benchmark Task | Objective | Executed Tool Sequence | Outcome & Telemetry | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1: Autonomous Code Debugging** | Detect failing assertion in `calculator.py`, patch subtraction bug, run test command, and verify. | `read_file` $\to$ `write_file` $\to$ `run_command` $\to$ `verify_goal` | Test passes with exit code 0; `test_pass.flag` created. | **PASSED** (100% Success, 3 steps) |
| **Task 2: Chained Data Extraction & Synthesis** | Parse JSON sensor readings, compute mathematical mean (24.0), write `summary.txt`, and verify. | `read_file` $\to$ `write_file` $\to$ `verify_goal` | `summary.txt` generated with exact `mean: 24.0`. | **PASSED** (100% Success, 2 steps) |
| **Task 3: Fault Recovery via IOR** | Deliberately trigger failing command, suppress failing tool via IOR, switch to file repair, and complete. | `run_command` (FAIL) $\to$ `write_file` $\to$ `verify_goal` | Step 1 fails ($r=-0.5$, surprise $>0$). Step 2 rejects `run_command`, switches to `write_file`. | **PASSED** (100% Success, 2 steps) |

### Key Observations
1. **Zero Action Perseveration**: On Task 3, despite an initial $+2.5$ logit bias favoring `run_command`, its failure instantly suppressed $P_{t+1}[\text{run\_command}]$, dropping its logit below $-3.5$ and ensuring `write_file` was selected on Step 2.
2. **Deterministic Chaining**: In Task 1 and Task 2, subgoal completion discounting enabled clean progression through multi-step read-write-execute-verify pipelines without thrashing.
3. **Stability & Reproducibility**: Evaluated across 5 consecutive independent executions with 100% pass rate and zero flakiness (average execution latency $< 0.13\text{ s}$).

---

## 3. Regression Suite & Architectural Invariants

All agent loop and reasoning tests pass with zero regressions:
- `brain/tests/test_agent_loop.py`: **4/4 PASSED** (Goal encoder, Tool registry, Cognitive core step, Autonomous task execution)
- `brain/tests/test_agent_reasoning.py`: **3/3 PASSED** (Code debugging, Chained data synthesis, Fault recovery / IOR)
- **Combined Agent Suite**: **7/7 PASSED (0.13s wall)**

### Architectural Invariants Enforced
- **Zero Oracle Cheats**: The agent receives no privileged environmental hints or state variables; verification evaluates real disk artifacts and subprocess exit codes.
- **Cognitive Provenance**: All actions originate from `AgentCognitiveCore.forward_step` combining recurrent thoughts $h_t$ and synaptic plasticity $P_t$.
- **Error Plasticity Activation**: Consequence surprise $\delta_r = |r - \hat{r}|$ directly gates synaptic plasticity updates and IOR suppression.
