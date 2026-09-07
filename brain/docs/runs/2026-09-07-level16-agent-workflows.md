# Level 16 Autonomous Agent Workflows: Multi-File Software Engineering, Code Inspection, and Git Workflows

**Date:** 2026-09-07  
**Platform:** Local Windows CPU (Python 3.11 deterministic runtime)  
**Status:** COMPLETE & FULLY VERIFIED (100% Pass across Level 16 Benchmark and Full Agent Regression Suite)  
**Implementation:** `brain/src/irene_brain/agent/tools.py`, `brain/src/irene_brain/agent/__init__.py`, `brain/src/irene_brain/agent/loop.py`  
**Test Suites:** `brain/tests/test_level16_agent_workflows.py`, `brain/tests/test_agent_reasoning.py`, `brain/tests/test_agent_loop.py`  

---

## 1. Context & Mechanistic Objective

Following the Master Autonomous Roadmap (`brain/docs/MASTER_ROADMAP.md`), the Pseudo-Brain cognitive agent must scale from micro-tasks (Level 15: single-file isolated reads and writes) to full multi-file software engineering (Level 16). Real-world codebases require recursive repository navigation, regex-based symbol cross-referencing, safe selective line patching without destructive overwrites, version control operations (branching, staging, committing), and autonomous fault recovery under failed tool outcomes.

### Level 15 Architectural Bottlenecks
1. **Single-File Scope**: The agent could only read or overwrite explicit file paths given in advance; it lacked capabilities to search across directory trees or discover relevant modules autonomously.
2. **Destructive Whole-File Rewrites**: The agent only had `write_file`, requiring rewriting entire multi-thousand-line files to change a single line, causing high token overhead and risking collateral corruption.
3. **No Version Control Awareness**: The agent could not inspect working tree changes, manage feature branches, or stage and commit atomic changes into git.

### Level 16 Engineering Tooling Architecture
To overcome these limitations, `brain/src/irene_brain/agent/tools.py` was expanded with modular, high-performance tools:

1. **`FileGrepTool` (`grep_files`)**:
   - Fast recursive regex and substring search across directories and files.
   - Automatically skips non-source noise directories (`.git`, `__pycache__`, `.venv`, `node_modules`).
   - Supports file extension filters (`extension=".py"` or `extensions=[...]`), case sensitivity toggles, and maximum match bounds.
   - Outputs line-numbered match records (`{rel_path}:{line_no}: {line}`) with positive reward feedback.

2. **`DirectoryListTool` (`list_directory`)**:
   - Recursive structural inspection with configurable `max_depth` limits.
   - Filters entries by file extensions and annotates file sizes in bytes.
   - Distinguishes `[DIR]` and `[FILE]` entries formatted with normalized posix paths.

3. **`FilePatchTool` (`patch_file`)**:
   - Surgical selective editing: replaces exact targeted lines (by 1-indexed `line_number` and/or `target_text`) or unique substring occurrences without altering surrounding code.
   - Enforces strict boundary and content validation: returns descriptive error results and negative reward if the target text is missing or ambiguous, preventing unintended corruptions.

4. **Git Operations (`GitStatusTool`, `GitCommitTool`, `GitBranchTool`)**:
   - `GitStatusTool` (`git_status`): Safe subprocess query capturing active branch (`git rev-parse --abbrev-ref HEAD`) and porcelain working tree modifications.
   - `GitCommitTool` (`git_commit`): Stages specified files or all working changes (`git add -A`) and commits with structured messages.
   - `GitBranchTool` (`git_branch`): Creates and switches git branches (`git checkout -b <branch>`).

5. **`ToolRegistry` Level 16 Extensions**:
   - Factory methods `create_level16_registry()` and `create_default()` providing clean preset configurations.
   - Inspection and membership introspection: `has_tool()`, `get_tool_names()`, `describe_tools()`, `__contains__`, and `__iter__`.

---

## 2. Empirical Benchmark Battery [MEASURED]

The Level 16 agent architecture was rigorously evaluated across 3 comprehensive autonomous engineering workflows plus 5 tool unit test suites in `brain/tests/test_level16_agent_workflows.py`:

| Benchmark Workflow | Objective | Executed Cognitive Progression | Telemetry & Results | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Benchmark 1: Multi-File Bug Investigation & Selective Patching** | Locate an off-by-one formula bug in `math_pkg/stats.py` across multiple modules, inspect the culprit file, patch the exact faulty line without modifying surrounding functions, run unit tests, and verify 100% pass. | `grep_files` $\to$ `read_file` $\to$ `patch_file` $\to$ `run_command` $\to$ `verify_goal` | - Step 1 (`grep_files`): located `stats.py:6: def sample_variance`<br>- Step 2 (`read_file`): inspected function body<br>- Step 3 (`patch_file`): patched denominator to `len(data) - 1`<br>- Step 4 (`run_command`): executed `unittest tests/test_stats.py` (exit 0)<br>- Step 5 (`verify_goal`): `test_stats.passed` flag created | **PASSED** (100% Success, 5 steps) |
| **Benchmark 2: Autonomous Test Suite Generation** | Discover an untested utility module `lib/string_ops.py` via directory inspection, inspect signatures (`slugify`, `truncate`, `camel_to_snake`), synthesize a complete unit test suite `tests/test_string_ops.py`, run tests, and verify pass. | `list_directory` $\to$ `read_file` $\to$ `write_file` $\to$ `run_command` $\to$ `verify_goal` | - Step 1 (`list_directory`): identified `string_ops.py` in `lib/`<br>- Step 2 (`read_file`): extracted API definitions<br>- Step 3 (`write_file`): generated `test_string_ops.py`<br>- Step 4 (`run_command`): ran test suite with exit 0<br>- Step 5 (`verify_goal`): 100% test assertion satisfaction confirmed | **PASSED** (100% Success, 5 steps) |
| **Benchmark 3: Git Branch Workflow & Fault Recovery** | Handle an intentionally failing git operation on Step 1, trigger consequence surprise, utilize Inhibition of Return (IOR) to prevent command perseveration, switch to feature branch, stage/commit changes, inspect status, and verify. | `run_command` (FAIL) $\to$ `git_branch` $\to$ `write_file` $\to$ `git_commit` $\to$ `verify_goal` | - Step 1 (`run_command`): failed on invalid push ($r=-0.1$, surprise $\delta_r > 0$)<br>- IOR suppression: $P_{t+1}[\text{run\_command}]$ dropped by $-6.0$<br>- Step 2 (`git_branch`): switched to `feat/autonomous-patch`<br>- Step 3 (`write_file`): created `feature.py`<br>- Step 4 (`git_commit`): committed changes<br>- Step 5 (`verify_goal`): confirmed clean git log & branch | **PASSED** (100% Success, 5 steps) |

### Tool-Level Unit Tests (5/5 PASSED)
- `test_file_grep_tool`: Regex matching, substring matching, extension filtering (`.py`), and non-existent path handling.
- `test_directory_list_tool`: Recursive directory traversal, `max_depth` limiting, file size recording, and posix path formatting.
- `test_file_patch_tool`: Line-number selective patching, unique substring replacement, and error reporting on missing text.
- `test_git_tools_basic`: `git_status`, `git_commit`, and `git_branch` execution against an isolated repository.
- `test_tool_registry_level16`: Level 16 factory instantiation, tool lookup, and capability description queries.

---

## 3. Cognitive Dynamics & Inhibition of Return (IOR) Analysis

### Consequence Surprise and Motor Inhibition
In Benchmark 3, the agent was biased toward `run_command` ($+3.5$ bias) on Step 1. The executed command was deliberately invalid (`git push origin non_existent_branch_fails`), returning exit code 128.
1. **Surprise Computation**:
   $$\delta_r = |r_{\text{true}} - \hat{r}| = |-0.1 - \hat{r}| > 0$$
2. **Synaptic Suppression**:
   $$\text{Suppression}[0, a_{\text{run\_command}}] = -4.0 \cdot (\delta_r + 1.0) \le -4.0$$
3. **Action Selection Shift**:
   At step 2, `run_command` logit fell well below $0$, completely eliminating perseverative retry loops and forcing the policy head to select `git_branch` ($+2.5$ bias), cleanly transitioning from failure to recovery.

### Subgoal Completion Discounting
On successful actions ($r > 0$, `last_success is True`), completion discounting applies:
$$\text{Discount}[0, a_{\text{completed}}] = -1.2$$
This enables seamless cognitive handoffs along multi-tool pipelines (e.g. `grep_files` $\to$ `read_file` $\to$ `patch_file` $\to$ `run_command` $\to$ `verify_goal`) without hardcoded loops or external supervisor intervention.

---

## 4. Test Suite Execution & Regression Invariants

### Execution Command
```bash
cmd /c "set PYTHONPATH=brain/src && py -3.11 -m unittest brain/tests/test_level16_agent_workflows.py"
```

### Output Telemetry
```
Ran 8 tests in 1.177s

OK
```

### Full Agent Regression Matrix
- `brain/tests/test_level16_agent_workflows.py`: **8/8 PASSED** (All Level 16 workflows & tool unit tests)
- `brain/tests/test_agent_reasoning.py`: **3/3 PASSED** (Level 15 reasoning & IOR regression)
- `brain/tests/test_agent_loop.py`: **4/4 PASSED** (Core cognitive recurrent step & autonomous task execution)
- **Total Autonomous Agent Suite**: **15/15 PASSED (100% Pass Rate across all levels)**

### Architectural Invariants Enforced
- **Selective Non-Destructive Editing**: File edits via `FilePatchTool` preserve existing lines, comments, and structure intact.
- **Zero Hallucination / Grounded Disk Artifacts**: All test verifications check actual on-disk files, git commit logs, and subprocess return codes.
- **Adaptive Resilience**: Command failures cleanly invoke synaptic plasticity and consequence surprise, redirecting policy flow to alternative repair actions.
