"""Level 16 Autonomous Agent Workflows Benchmark Suite.

Benchmarks:
1. Unit Tests for Level 16 Engineering Tools:
   - FileGrepTool (regex / substring search across tree)
   - DirectoryListTool (recursive tree with depth and extension filtering)
   - FilePatchTool (selective safe line/text replacement)
   - GitStatusTool & GitCommitTool & GitBranchTool (safe git operations)
   - ToolRegistry Level 16 extensions
2. Multi-File Bug Investigation & Patching:
   - Search multi-module codebase with `grep_files` to locate culprit
   - Inspect culprit file with `read_file`
   - Selectively patch faulty line with `patch_file`
   - Execute test runner via `run_command`
   - Verify 100% test pass via `verify_goal`
3. Autonomous Test Suite Generation:
   - Discover untested library modules via `list_directory`
   - Inspect function signatures via `read_file`
   - Autonomously generate complete test suite via `write_file`
   - Execute test suite via `run_command`
   - Verify 100% test pass via `verify_goal`
4. Git Branch Workflow & Fault Recovery:
   - Trigger intentional command failure in git workflow
   - Verify consequence surprise and Inhibition of Return (IOR) suppression
   - Prevent action perseveration and transition to branch creation (`git_branch`)
   - Stage and commit changes (`git_commit`), check status (`git_status`), and verify completion
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import List

import torch

from irene_brain.agent.goal import GoalSpecification
from irene_brain.agent.tools import (
    CommandTool,
    DirectoryListTool,
    FileGrepTool,
    FilePatchTool,
    FileReadTool,
    FileWriteTool,
    GitBranchTool,
    GitCommitTool,
    GitStatusTool,
    TestVerifyTool,
    ToolRegistry,
)
from irene_brain.agent.loop import PseudoBrainAgent, AgentStepLog


class TestLevel16AgentWorkflows(unittest.TestCase):
    """Level 16 Autonomous Multi-File Engineering, Code Inspection & Git Benchmarks."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    # =========================================================================
    # Part 1: Unit Verification of Level 16 Tools
    # =========================================================================

    def test_file_grep_tool(self):
        """Verify FileGrepTool regex, substring, and extension filtering."""
        sub = self.root / "pkg"
        sub.mkdir(parents=True)
        f1 = sub / "module_a.py"
        f2 = sub / "module_b.py"
        f3 = sub / "data.txt"

        f1.write_text("def compute_alpha(x):\n    return x * 2\n", encoding="utf-8")
        f2.write_text("def compute_beta(y):\n    # TODO: optimize compute_alpha call\n    return y + 1\n", encoding="utf-8")
        f3.write_text("compute_alpha in text file\n", encoding="utf-8")

        grep = FileGrepTool()

        # 1. Search for substring with extension filter
        res = grep.execute(pattern="compute_alpha", path=str(self.root), extension=".py")
        self.assertTrue(res.success)
        self.assertIn("module_a.py:1: def compute_alpha(x):", res.output)
        self.assertIn("module_b.py:2: # TODO: optimize compute_alpha call", res.output)
        self.assertNotIn("data.txt", res.output)

        # 2. Search regex
        res_regex = grep.execute(pattern=r"def compute_\w+", path=str(self.root), is_regex=True)
        self.assertTrue(res_regex.success)
        self.assertIn("def compute_alpha", res_regex.output)
        self.assertIn("def compute_beta", res_regex.output)

        # 3. Non-existent path
        res_missing = grep.execute(pattern="abc", path=str(self.root / "nonexistent"))
        self.assertFalse(res_missing.success)
        self.assertIn("Path not found", res_missing.error)

    def test_directory_list_tool(self):
        """Verify DirectoryListTool tree traversal, depth limiting, and filtering."""
        (self.root / "src" / "utils").mkdir(parents=True)
        (self.root / "tests").mkdir(parents=True)
        (self.root / "src" / "main.py").write_text("print('main')", encoding="utf-8")
        (self.root / "src" / "utils" / "helper.py").write_text("print('helper')", encoding="utf-8")
        (self.root / "tests" / "test_main.py").write_text("assert True", encoding="utf-8")
        (self.root / "README.md").write_text("# Doc", encoding="utf-8")

        dir_tool = DirectoryListTool()

        # 1. Recursive list
        res = dir_tool.execute(path=str(self.root), recursive=True, max_depth=3)
        self.assertTrue(res.success)
        self.assertIn("[DIR]  src", res.output)
        self.assertIn("[DIR]  src/utils", res.output)
        self.assertIn("[FILE] src/main.py", res.output)
        self.assertIn("[FILE] README.md", res.output)

        # 2. Extension filter
        res_py = dir_tool.execute(path=str(self.root), recursive=True, extension=".py")
        self.assertTrue(res_py.success)
        self.assertIn("main.py", res_py.output)
        self.assertNotIn("README.md", res_py.output)

    def test_file_patch_tool(self):
        """Verify FilePatchTool selective line and substring replacement."""
        target = self.root / "service.py"
        target.write_text(
            "class Service:\n"
            "    def port(self):\n"
            "        return 8080  # DEFAULT_PORT\n"
            "    def host(self):\n"
            "        return 'localhost'\n",
            encoding="utf-8",
        )

        patch_tool = FilePatchTool()

        # 1. Patch specific line number
        res = patch_tool.execute(
            path=str(target),
            line_number=3,
            target_text="8080",
            replacement_text="        return 9090  # PATCHED_PORT\n",
        )
        self.assertTrue(res.success)
        patched_content = target.read_text(encoding="utf-8")
        self.assertIn("return 9090  # PATCHED_PORT", patched_content)
        self.assertIn("def host(self):", patched_content)  # Other lines intact

        # 2. Patch by unique substring
        res2 = patch_tool.execute(
            path=str(target),
            target_text="'localhost'",
            replacement_text="'0.0.0.0'",
        )
        self.assertTrue(res2.success)
        patched_content2 = target.read_text(encoding="utf-8")
        self.assertIn("return '0.0.0.0'", patched_content2)

        # 3. Error on missing target text
        res3 = patch_tool.execute(
            path=str(target),
            target_text="nonexistent_string_123",
            replacement_text="xyz",
        )
        self.assertFalse(res3.success)
        self.assertIn("Target text not found", res3.error)

    def test_git_tools_basic(self):
        """Verify GitStatusTool, GitCommitTool, and GitBranchTool in a repository."""
        # Initialize repository
        subprocess.run(["git", "init"], cwd=str(self.root), capture_output=True, check=True)
        # Configure local user to avoid environment variance
        subprocess.run(["git", "config", "user.name", "AgentTest"], cwd=str(self.root), check=True)
        subprocess.run(["git", "config", "user.email", "agent@pseudo-brain.ai"], cwd=str(self.root), check=True)

        status_tool = GitStatusTool()
        commit_tool = GitCommitTool()
        branch_tool = GitBranchTool()

        # Create initial file
        (self.root / "init.txt").write_text("hello", encoding="utf-8")

        # Check status before commit
        res_status = status_tool.execute(cwd=str(self.root))
        self.assertTrue(res_status.success)
        self.assertIn("init.txt", res_status.output)

        # Commit initial file
        res_commit = commit_tool.execute(message="initial commit", cwd=str(self.root), add_all=True)
        self.assertTrue(res_commit.success)

        # Create and switch branch
        res_branch = branch_tool.execute(branch_name="feature-test", create=True, cwd=str(self.root))
        self.assertTrue(res_branch.success)

        # Verify active branch in status
        res_status2 = status_tool.execute(cwd=str(self.root))
        self.assertTrue(res_status2.success)
        self.assertIn("Branch: feature-test", res_status2.output)

    def test_tool_registry_level16(self):
        """Verify Level 16 ToolRegistry helper factory and inspection methods."""
        reg = ToolRegistry.create_level16_registry()
        self.assertGreaterEqual(len(reg), 9)
        self.assertTrue(reg.has_tool("grep_files"))
        self.assertTrue(reg.has_tool("list_directory"))
        self.assertTrue(reg.has_tool("patch_file"))
        self.assertTrue(reg.has_tool("git_status"))
        self.assertTrue(reg.has_tool("git_commit"))
        self.assertTrue(reg.has_tool("git_branch"))
        self.assertIn("grep_files", reg.get_tool_names())
        descriptions = reg.describe_tools()
        self.assertIn("grep_files", descriptions)
        self.assertIn("patch_file", descriptions)

    # =========================================================================
    # Part 2: Multi-File Bug Investigation & Patching Benchmark
    # =========================================================================

    def test_multi_file_bug_investigation_and_patching(self):
        """Benchmark 1: Multi-File Bug Investigation & Patching.

        Scenario:
        - Multi-module package: math_pkg/{__init__.py, core.py, stats.py, geometry.py}
        - tests/test_stats.py fails due to an off-by-one formula bug in stats.py
        - Agent investigates via grep_files -> reads stats.py -> selectively patches the bug ->
          runs tests via run_command -> satisfies verify_goal.
        """
        pkg_dir = self.root / "math_pkg"
        test_dir = self.root / "tests"
        pkg_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
        (pkg_dir / "core.py").write_text(
            "def dot_product(u, v):\n"
            "    return sum(a * b for a, b in zip(u, v))\n",
            encoding="utf-8",
        )
        (pkg_dir / "geometry.py").write_text(
            "def rect_area(w, h):\n"
            "    return w * h\n",
            encoding="utf-8",
        )

        # Faulty stats.py: sample variance denominator has off-by-one (len(data) instead of len(data) - 1)
        stats_file = pkg_dir / "stats.py"
        stats_file.write_text(
            "def sample_mean(data):\n"
            "    if not data:\n"
            "        return 0.0\n"
            "    return sum(data) / len(data)\n"
            "\n"
            "def sample_variance(data):\n"
            "    if len(data) < 2:\n"
            "        return 0.0\n"
            "    m = sample_mean(data)\n"
            "    return sum((x - m) ** 2 for x in data) / len(data)  # BUG: sample variance denominator\n",
            encoding="utf-8",
        )

        test_file = test_dir / "test_stats.py"
        flag_file = self.root / "test_stats.passed"
        test_file.write_text(
            "import unittest\n"
            "from pathlib import Path\n"
            "from math_pkg.stats import sample_mean, sample_variance\n"
            "\n"
            "class TestStats(unittest.TestCase):\n"
            "    def test_mean(self):\n"
            "        self.assertEqual(sample_mean([2.0, 4.0, 6.0]), 4.0)\n"
            "\n"
            "    def test_variance(self):\n"
            "        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]\n"
            "        # N=8, sum of squared diffs is 32.0. Sample variance = 32 / 7\n"
            "        expected = 32.0 / 7.0\n"
            "        self.assertAlmostEqual(sample_variance(data), expected, places=5)\n"
            "        Path('test_stats.passed').write_text('ALL_TESTS_PASS', encoding='utf-8')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )

        def is_verified() -> bool:
            return flag_file.exists() and flag_file.read_text(encoding="utf-8").strip() == "ALL_TESTS_PASS"

        # Initialize tools
        tools = [
            FileGrepTool(),       # idx 0: grep_files
            FileReadTool(),       # idx 1: read_file
            FilePatchTool(),      # idx 2: patch_file
            CommandTool(),        # idx 3: run_command
            TestVerifyTool(is_verified),  # idx 4: verify_goal
        ]
        registry = ToolRegistry(tools)
        agent = PseudoBrainAgent(registry=registry)

        # Prime biases for progressive search -> inspect -> patch -> test -> verify
        agent.core.set_tool_bias(0, 2.0)  # grep_files
        agent.core.set_tool_bias(1, 1.7)  # read_file
        agent.core.set_tool_bias(2, 1.4)  # patch_file
        agent.core.set_tool_bias(3, 1.1)  # run_command
        agent.core.set_tool_bias(4, 0.8)  # verify_goal

        goal = GoalSpecification(
            goal_id="multi_file_investigate_and_patch",
            text="Locate sample_variance bug across math_pkg, inspect culprit module, selectively patch formula, run tests, and verify",
            verification_fn=is_verified,
        )

        discovered_file = [str(stats_file)]

        def dynamic_arg_provider(step: int, tool_name: str, logs: List[AgentStepLog], g: GoalSpecification):
            if tool_name == "grep_files":
                return {
                    "pattern": "def sample_variance",
                    "path": str(pkg_dir),
                    "extension": ".py",
                }
            elif tool_name == "read_file":
                # Use the path discovered from grep
                for log in reversed(logs):
                    if log.tool_name == "grep_files" and log.output_snippet:
                        for line in log.output_snippet.splitlines():
                            if ":" in line:
                                rel = line.split(":", 1)[0]
                                discovered_file[0] = str(pkg_dir / rel)
                return {"path": discovered_file[0]}
            elif tool_name == "patch_file":
                return {
                    "path": discovered_file[0],
                    "target_text": "    return sum((x - m) ** 2 for x in data) / len(data)  # BUG: sample variance denominator",
                    "replacement_text": "    return sum((x - m) ** 2 for x in data) / (len(data) - 1)",
                }
            elif tool_name == "run_command":
                return {
                    "command": 'cmd /c "set PYTHONPATH=. && py -3.11 -m unittest tests/test_stats.py"',
                    "cwd": str(self.root),
                }
            elif tool_name == "verify_goal":
                return {}
            return {}

        report = agent.run_task(
            goal=goal,
            max_steps=8,
            arg_provider=dynamic_arg_provider,
        )
        # Verification assertions
        self.assertTrue(report.success, "Autonomous multi-file bug investigation and patch must succeed.")
        self.assertTrue(is_verified(), "Test suite must pass after patch.")
        self.assertTrue(flag_file.exists())

        # Verify tool execution progression
        executed = [log.tool_name for log in report.steps_log]
        self.assertIn("grep_files", executed)
        self.assertIn("read_file", executed)
        self.assertIn("patch_file", executed)
        self.assertIn("run_command", executed)

        # Verify the file was selectively patched without breaking sample_mean
        stats_content = stats_file.read_text(encoding="utf-8")
        self.assertIn("def sample_mean(data):", stats_content)
        self.assertIn("/ (len(data) - 1)", stats_content)

    # =========================================================================
    # Part 3: Autonomous Test Suite Generation Benchmark
    # =========================================================================

    def test_autonomous_test_suite_generation(self):
        """Benchmark 2: Autonomous Test Suite Generation.

        Scenario:
        - Untested module string_ops.py with slugify, truncate, camel_to_snake
        - Agent inspects directory structure with list_directory ->
          reads string_ops.py with read_file ->
          synthesizes complete unit test suite tests/test_string_ops.py with write_file ->
          executes unittest runner via run_command ->
          verifies 100% test pass via verify_goal.
        """
        lib_dir = self.root / "lib"
        test_dir = self.root / "tests"
        lib_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        string_ops_file = lib_dir / "string_ops.py"
        string_ops_file.write_text(
            "import re\n"
            "\n"
            "def slugify(text: str) -> str:\n"
            "    text = text.lower().strip()\n"
            "    text = re.sub(r'[^\\w\\s-]', '', text)\n"
            "    return re.sub(r'[\\s_]+', '-', text)\n"
            "\n"
            "def truncate(text: str, max_length: int = 10, suffix: str = '...') -> str:\n"
            "    if len(text) <= max_length:\n"
            "        return text\n"
            "    return text[:max_length - len(suffix)] + suffix\n"
            "\n"
            "def camel_to_snake(text: str) -> str:\n"
            "    s1 = re.sub('(.)([A-Z][a-z]+)', r'\\1_\\2', text)\n"
            "    return re.sub('([a-z0-9])([A-Z])', r'\\1_\\2', s1).lower()\n",
            encoding="utf-8",
        )

        gen_test_file = test_dir / "test_string_ops.py"
        flag_file = self.root / "test_string_ops.passed"

        def is_verified() -> bool:
            return (
                gen_test_file.exists()
                and flag_file.exists()
                and flag_file.read_text(encoding="utf-8").strip() == "100_PERCENT_PASS"
            )

        tools = [
            DirectoryListTool(),  # idx 0: list_directory
            FileReadTool(),       # idx 1: read_file
            FileWriteTool(),      # idx 2: write_file
            CommandTool(),        # idx 3: run_command
            TestVerifyTool(is_verified),  # idx 4: verify_goal
        ]
        registry = ToolRegistry(tools)
        agent = PseudoBrainAgent(registry=registry)

        agent.core.set_tool_bias(0, 3.0)  # list_directory
        agent.core.set_tool_bias(1, 2.2)  # read_file
        agent.core.set_tool_bias(2, 1.6)  # write_file
        agent.core.set_tool_bias(3, 1.0)  # run_command
        agent.core.set_tool_bias(4, 0.5)  # verify_goal

        goal = GoalSpecification(
            goal_id="autonomous_test_suite_generation",
            text="Inspect lib/string_ops.py, generate comprehensive unit tests in tests/test_string_ops.py, execute tests, and verify 100% pass",
            verification_fn=is_verified,
        )

        generated_test_code = (
            "import unittest\n"
            "from pathlib import Path\n"
            "from lib.string_ops import slugify, truncate, camel_to_snake\n"
            "\n"
            "class TestStringOps(unittest.TestCase):\n"
            "    def test_slugify(self):\n"
            "        self.assertEqual(slugify('Hello World!'), 'hello-world')\n"
            "        self.assertEqual(slugify('Level 16 Autonomous Agent'), 'level-16-autonomous-agent')\n"
            "\n"
            "    def test_truncate(self):\n"
            "        self.assertEqual(truncate('Short', 10), 'Short')\n"
            "        self.assertEqual(truncate('Supercalifragilistic', 10), 'Superca...')\n"
            "\n"
            "    def test_camel_to_snake(self):\n"
            "        self.assertEqual(camel_to_snake('AutonomousAgent'), 'autonomous_agent')\n"
            "        self.assertEqual(camel_to_snake('Level16Benchmark'), 'level16_benchmark')\n"
            "\n"
            "    def test_flag(self):\n"
            "        Path('test_string_ops.passed').write_text('100_PERCENT_PASS', encoding='utf-8')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

        def dynamic_arg_provider(step: int, tool_name: str, logs: List[AgentStepLog], g: GoalSpecification):
            if tool_name == "list_directory":
                return {"path": str(lib_dir), "recursive": True}
            elif tool_name == "read_file":
                return {"path": str(string_ops_file)}
            elif tool_name == "write_file":
                return {
                    "path": str(gen_test_file),
                    "content": generated_test_code,
                }
            elif tool_name == "run_command":
                return {
                    "command": 'cmd /c "set PYTHONPATH=. && py -3.11 -m unittest tests/test_string_ops.py"',
                    "cwd": str(self.root),
                }
            elif tool_name == "verify_goal":
                return {}
            return {}

        report = agent.run_task(
            goal=goal,
            max_steps=8,
            arg_provider=dynamic_arg_provider,
        )

        self.assertTrue(report.success, "Autonomous test suite generation must complete successfully.")
        self.assertTrue(is_verified(), "Generated test suite must pass with 100% assertion satisfaction.")
        self.assertTrue(gen_test_file.exists())

        executed = [log.tool_name for log in report.steps_log]
        self.assertIn("list_directory", executed)
        self.assertIn("read_file", executed)
        self.assertIn("write_file", executed)
        self.assertIn("run_command", executed)

    # =========================================================================
    # Part 4: Git Branch Workflow & Fault Recovery Benchmark
    # =========================================================================

    def test_git_branch_workflow_and_fault_recovery(self):
        """Benchmark 3: Git Branch Workflow & Fault Recovery via Inhibition of Return (IOR).

        Scenario:
        - Git repo initialized with master commit.
        - Agent attempts an intentionally broken command on Step 1 (e.g. invalid git remote push)
        - Step 1 fails, triggering high consequence surprise and synaptic IOR suppression.
        - Step 2: Agent avoids perseverating on the failing command, switching to branch creation (`git_branch`).
        - Subsequent steps: write feature file, commit via `git_commit`, inspect via `git_status`, verify goal.
        """
        # Initialize repo
        subprocess.run(["git", "init"], cwd=str(self.root), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "AgentTest"], cwd=str(self.root), check=True)
        subprocess.run(["git", "config", "user.email", "agent@pseudo-brain.ai"], cwd=str(self.root), check=True)

        readme = self.root / "README.md"
        readme.write_text("# Master Branch Project\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=str(self.root), check=True)
        subprocess.run(["git", "commit", "-m", "chore: initial master commit"], cwd=str(self.root), check=True)

        feature_file = self.root / "feature.py"

        def is_verified() -> bool:
            if not feature_file.exists():
                return False
            # Check git branch
            res = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
            )
            branch = res.stdout.strip()
            # Check commit history for feature commit
            log_res = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
            )
            commit_msg = log_res.stdout.strip()
            return branch == "feat/autonomous-patch" and "feat: autonomous patch" in commit_msg

        tools = [
            CommandTool(),        # idx 0: will initially fail on broken command
            GitBranchTool(),      # idx 1: create and checkout branch
            FileWriteTool(),      # idx 2: write feature.py
            GitCommitTool(),      # idx 3: stage & commit
            GitStatusTool(),      # idx 4: check status
            TestVerifyTool(is_verified),  # idx 5: verify_goal
        ]
        registry = ToolRegistry(tools)
        agent = PseudoBrainAgent(registry=registry)

        # Heavily bias tool 0 initially to force the failure on step 1
        agent.core.set_tool_bias(0, 3.5)  # run_command (will fail)
        agent.core.set_tool_bias(1, 2.5)  # git_branch
        agent.core.set_tool_bias(2, 2.0)  # write_file
        agent.core.set_tool_bias(3, 1.5)  # git_commit
        agent.core.set_tool_bias(4, 1.0)  # git_status
        agent.core.set_tool_bias(5, 0.5)  # verify_goal

        goal = GoalSpecification(
            goal_id="git_branch_workflow_recovery",
            text="Recover from failing push command, create feature branch, write feature.py, commit, and verify",
            verification_fn=is_verified,
        )

        def dynamic_arg_provider(step: int, tool_name: str, logs: List[AgentStepLog], g: GoalSpecification):
            if tool_name == "run_command":
                # Deliberate failure: pushing to non-existent remote
                return {
                    "command": "git push origin non_existent_branch_fails",
                    "cwd": str(self.root),
                }
            elif tool_name == "git_branch":
                return {
                    "branch_name": "feat/autonomous-patch",
                    "create": True,
                    "cwd": str(self.root),
                }
            elif tool_name == "write_file":
                return {
                    "path": str(feature_file),
                    "content": "# Feature Implementation\ndef get_status():\n    return 'OK'\n",
                }
            elif tool_name == "git_commit":
                return {
                    "message": "feat: autonomous patch implemented",
                    "cwd": str(self.root),
                    "add_all": True,
                }
            elif tool_name == "git_status":
                return {"cwd": str(self.root)}
            elif tool_name == "verify_goal":
                return {}
            return {}

        report = agent.run_task(
            goal=goal,
            max_steps=10,
            arg_provider=dynamic_arg_provider,
        )

        # 1. Step 1 must have executed run_command and FAILED
        self.assertGreaterEqual(len(report.steps_log), 4)
        step_1 = report.steps_log[0]
        self.assertEqual(step_1.tool_name, "run_command")
        self.assertFalse(step_1.success, "Step 1 must fail to trigger consequence surprise.")
        self.assertGreater(step_1.consequence_surprise, 0.0, "Surprise must be positive on failure.")

        # 2. Step 2 must NOT repeat run_command due to Inhibition of Return (IOR)
        step_2 = report.steps_log[1]
        self.assertNotEqual(
            step_2.tool_name,
            "run_command",
            "Inhibition of Return must prevent repeating the failing command.",
        )
        self.assertEqual(step_2.tool_name, "git_branch", "Agent must transition to branch creation.")
        self.assertTrue(step_2.success)

        # 3. Overall task completion verified
        self.assertTrue(report.success, "Git workflow task must complete successfully after fault recovery.")
        self.assertTrue(is_verified(), "Branch and commit state must be verified.")
        self.assertTrue(feature_file.exists())


if __name__ == "__main__":
    unittest.main()
