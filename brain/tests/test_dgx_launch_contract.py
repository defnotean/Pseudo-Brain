from __future__ import annotations

from hashlib import sha256
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


class DgxLaunchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.brain_root = Path(__file__).resolve().parents[1]
        cls.script_root = cls.brain_root / "scripts" / "dgx"
        cls.common = (cls.script_root / "Dgx.Common.ps1").read_text(encoding="utf-8")
        cls.remote = (cls.script_root / "_remote_dispatch.sh").read_text(encoding="utf-8")

    def test_ssh_is_noninteractive_strict_and_remote_only(self) -> None:
        self.assertIn("BatchMode=yes", self.common)
        self.assertIn("StrictHostKeyChecking=yes", self.common)
        self.assertIn("IsLoopback", self.common)
        self.assertIn("refusing local GPU execution", self.common)
        self.assertIn("could not be proven to be a remote ARM64 DGX Spark", self.common)
        self.assertIn('.Replace("`r`n", "`n")', self.common)
        self.assertIn("PowerShell native-pipe newline sentinel", self.common)

    def test_workspace_v2_is_write_only_and_legacy_access_is_explicit(self) -> None:
        status = (self.script_root / "Get-DgxBrainTrainingStatus.ps1").read_text(
            encoding="utf-8"
        )
        retrieval = (self.script_root / "Receive-DgxBrainArtifact.ps1").read_text(
            encoding="utf-8"
        )
        for required in (
            "PseudoBrainWrite",
            "HistoricalIreneRead",
            "pseudo-brain",
            "historical Irene workspace is read-only",
            "direct child of the remote 'projects' directory",
            "cannot be nested under the historical Irene workspace",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.common)
        for required in (
            ".pseudo-brain-workspace-v2",
            "pseudo-brain-workspace-v2",
            "resolve_historical_workspace",
            "assert_owned_historical_workspace",
            "status|historical_status",
            "artifact|historical_artifact",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.remote)
        self.assertIn("$HistoricalIreneWorkspace", status)
        self.assertIn("historical_status", status)
        self.assertIn("$HistoricalIreneWorkspace", retrieval)
        self.assertIn("historical_artifact", retrieval)
        for filename in (
            "Invoke-DgxPreflight.ps1",
            "Sync-DgxBrainRelease.ps1",
            "Invoke-DgxBrainSmoke.ps1",
            "Start-DgxBrainTraining.ps1",
            "Resume-DgxBrainTraining.ps1",
        ):
            launcher = (self.script_root / filename).read_text(encoding="utf-8")
            with self.subTest(write_launcher=filename):
                self.assertNotIn("HistoricalIreneWorkspace", launcher)
                self.assertIn("Assert-DgxRemoteWorkDir", launcher)

    def test_workspace_resolvers_and_markers_reject_cross_generation_writes(self) -> None:
        body = """
PATH=/usr/bin:/bin
root="$(realpath -m -- "$1")/projects"
new="$root/pseudo-brain"
legacy="$root/irene-brain"
nested="$legacy/projects/pseudo-brain"
mkdir -p -- "$new" "$legacy" "$nested"
[[ "$(resolve_workspace "$new")" == "$(realpath -m -- "$new")" ]]
if (resolve_workspace "$legacy" >/dev/null 2>&1); then
    exit 101
fi
if (resolve_workspace "$nested" >/dev/null 2>&1); then
    exit 102
fi
[[ "$(resolve_historical_workspace "$legacy")" == "$(realpath -m -- "$legacy")" ]]
if (resolve_historical_workspace "$new" >/dev/null 2>&1); then
    exit 103
fi
printf '%s\n' pseudo-brain-workspace-v2 > "$new/.pseudo-brain-workspace-v2"
printf '%s\n' irene-brain-workspace-v1 > "$legacy/.irene-brain-workspace-v1"
assert_owned_workspace "$new"
assert_owned_historical_workspace "$legacy"
if (assert_owned_workspace "$legacy" >/dev/null 2>&1); then
    exit 104
fi
"""
        self._run_bash_function_contract(body)

    def test_canonical_rcq_workspace_and_authority_trees_require_private_account_state(self) -> None:
        body = """
PATH=/usr/bin:/bin
current_uid="$(id -u)"
test_home="$(realpath -m -- "$1/account-home")"
workspace="$test_home/projects/pseudo-brain"
mkdir -p -- "$workspace"
chmod 700 "$test_home" "$test_home/projects" "$workspace"
printf '%s\n' pseudo-brain-workspace-v2 > "$workspace/.pseudo-brain-workspace-v2"
chmod 600 "$workspace/.pseudo-brain-workspace-v2"
getent() {
    [[ "$1" == passwd && "$2" == "$current_uid" ]] || return 1
    printf 'codex:x:%s:%s::%s:/bin/bash\n' "$current_uid" "$current_uid" "$test_home"
}

[[ "$(resolve_canonical_rcq_v2_workspace)" == "$workspace" ]]
ln -- "$workspace/.pseudo-brain-workspace-v2" "$workspace/marker-hardlink"
if (resolve_canonical_rcq_v2_workspace >/dev/null 2>&1); then
    exit 101
fi
rm -- "$workspace/marker-hardlink"
if [[ "$OSTYPE" != msys* && "$OSTYPE" != cygwin* ]]; then
    chmod 620 "$workspace/.pseudo-brain-workspace-v2"
    if (resolve_canonical_rcq_v2_workspace >/dev/null 2>&1); then
        exit 102
    fi
    chmod 600 "$workspace/.pseudo-brain-workspace-v2"
fi

registry="$workspace/final-claims"
mkdir -- "$registry"
chmod 700 "$registry"
if [[ "$OSTYPE" != msys* && "$OSTYPE" != cygwin* ]]; then
    assert_account_owned_directory "$registry" "$workspace" 'claim registry' 700
    chmod 770 "$registry"
    if (assert_account_owned_directory "$registry" "$workspace" 'claim registry' 700 >/dev/null 2>&1); then
        exit 103
    fi
    chmod 700 "$registry"
else
    assert_account_owned_directory "$registry" "$workspace" 'claim registry'
fi

release="$workspace/releases/release-1"
mkdir -p -- "$release/brain"
printf '%s\n' safe > "$release/brain/file.py"
chmod 700 "$workspace/releases" "$release" "$release/brain"
chmod 600 "$release/brain/file.py"
assert_account_owned_tree "$release" "$workspace/releases" 'release tree'
if [[ "$OSTYPE" != msys* && "$OSTYPE" != cygwin* ]]; then
    chmod 620 "$release/brain/file.py"
    if (assert_account_owned_tree "$release" "$workspace/releases" 'release tree' >/dev/null 2>&1); then
        exit 104
    fi
    chmod 600 "$release/brain/file.py"
fi
ln -- "$release/brain/file.py" "$release/brain/hardlink.py"
if (assert_account_owned_tree "$release" "$workspace/releases" 'release tree' >/dev/null 2>&1); then
    exit 105
fi
"""
        self._run_bash_function_contract(body)

    def test_historical_artifact_action_is_read_only_and_write_action_rejects_root(self) -> None:
        bash = self._bash_path()
        bash_environment = dict(os.environ)
        bash_environment["PATH"] = "/usr/bin:/bin"
        with tempfile.TemporaryDirectory(prefix="dgx-historical-read-") as temporary:
            workspace = Path(temporary) / "projects" / "irene-brain"
            checkpoint_dir = workspace / "runs" / "legacy-run" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (workspace / ".irene-brain-workspace-v1").write_text(
                "irene-brain-workspace-v1\n",
                encoding="utf-8",
                newline="\n",
            )
            (checkpoint_dir / "step-00000001.pt").write_bytes(b"historical")
            before = self._tree_identity(workspace)

            retrieved = subprocess.run(
                [
                    bash,
                    str(self.script_root / "_remote_dispatch.sh"),
                    "historical_artifact",
                    self._git_bash_path(workspace),
                    "legacy-run",
                    "checkpoints",
                ],
                check=False,
                capture_output=True,
                env=bash_environment,
                text=True,
                timeout=15,
            )
            self.assertEqual(retrieved.returncode, 0, msg=retrieved.stderr)
            self.assertIn("artifact_path=", retrieved.stdout)
            self.assertEqual(self._tree_identity(workspace), before)

            rejected = subprocess.run(
                [
                    bash,
                    str(self.script_root / "_remote_dispatch.sh"),
                    "prepare_sync",
                    self._git_bash_path(workspace),
                    "1",
                ],
                check=False,
                capture_output=True,
                env=bash_environment,
                text=True,
                timeout=15,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("unsafe_workspace", rejected.stderr)
            self.assertEqual(self._tree_identity(workspace), before)

    def test_remote_launcher_never_pulls_or_stops_services(self) -> None:
        self.assertIn("--pull never", self.remote)
        self.assertNotIn("docker pull", self.remote)
        self.assertNotIn("docker stop", self.remote)
        self.assertNotIn("docker kill", self.remote)
        self.assertNotIn("systemctl", self.remote)

    def test_dispatch_and_generated_jobs_ignore_caller_path_shadows(self) -> None:
        for required in (
            "#!/bin/bash",
            "PATH='/usr/sbin:/usr/bin'",
            "unset BASH_ENV ENV PYTHONHOME PYTHONPATH CDPATH",
            "assert_trusted_host_runtime",
            "required command '$1' is not root-owned",
            "required command '$1' is group/other writable",
            "/usr/bin/env -i PATH=/usr/sbin:/usr/bin /bin/bash -p --noprofile --norc -s --",
            "-e PATH=/usr/sbin:/usr/bin",
            "--noprofile --norc -c",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.remote + self.common)
        self.assertNotIn("#!/usr/bin/env bash", self.remote)
        self.assertNotIn('"$image_id" -lc "$container_script"', self.remote)

        bash = self._bash_path()
        with tempfile.TemporaryDirectory(prefix="dgx-path-shadow-") as temporary:
            root = Path(temporary)
            workspace = root / "projects" / "irene-brain"
            checkpoint_dir = workspace / "runs" / "legacy-run" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (workspace / ".irene-brain-workspace-v1").write_text(
                "irene-brain-workspace-v1\n", encoding="utf-8", newline="\n"
            )
            (checkpoint_dir / "step-00000001.pt").write_bytes(b"historical")
            shadow = root / "shadow"
            shadow.mkdir()
            for name in ("realpath", "find", "sha256sum", "python3", "docker"):
                candidate = shadow / name
                candidate.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8", newline="\n")
                candidate.chmod(0o755)
            environment = dict(os.environ)
            environment["PATH"] = str(shadow) + os.pathsep + environment.get("PATH", "")
            completed = subprocess.run(
                [
                    bash,
                    str(self.script_root / "_remote_dispatch.sh"),
                    "historical_artifact",
                    self._git_bash_path(workspace),
                    "legacy-run",
                    "checkpoints",
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
                timeout=15,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertIn("artifact_path=", completed.stdout)

            hostile = root / "hostile-hooks"
            hostile.mkdir()
            bash_hook = hostile / "bash-env.sh"
            python_hook = hostile / "python-startup.py"
            marker = hostile / "hook-ran"
            bash_hook.write_text(
                f"printf 'bash-env\\n' > '{self._git_bash_path(marker)}'\nexit 99\n",
                encoding="utf-8",
                newline="\n",
            )
            python_hook.write_text(
                "open(r'%s', 'w', encoding='utf-8').write('python-startup\\n')\n"
                "raise SystemExit(99)\n" % marker.as_posix(),
                encoding="utf-8",
                newline="\n",
            )
            environment["BASH_ENV"] = str(bash_hook)
            environment["ENV"] = str(bash_hook)
            environment["PYTHONSTARTUP"] = str(python_hook)
            environment["PYTHONPATH"] = str(hostile)
            environment["PYTHONHOME"] = str(hostile)
            # Git Bash rejects combining -p with GNU long options. Linux DGX
            # uses both; this local check covers privileged BASH_ENV ignoring.
            hardened = subprocess.run(
                [
                    bash,
                    "-p",
                    str(self.script_root / "_remote_dispatch.sh"),
                    "historical_artifact",
                    self._git_bash_path(workspace),
                    "legacy-run",
                    "checkpoints",
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
                timeout=15,
            )
            self.assertEqual(hardened.returncode, 0, msg=hardened.stderr)
            self.assertIn("artifact_path=", hardened.stdout)
            self.assertFalse(marker.exists())

    def test_release_python_source_allowlist_rejects_import_shadows(self) -> None:
        body = """
PATH=/usr/bin:/bin
release="$1/release"
source_root="$release/brain/src"
package="$source_root/irene_brain"
mkdir -p -- "$package/evaluation"
printf '%s\n' 'PACKAGE = 1' > "$package/__init__.py"
printf '%s\n' 'VALUE = 1' > "$package/evaluation/gate.py"
assert_clean_release_python_source "$release"

printf '%s\n' 'raise SystemExit(99)' > "$source_root/sitecustomize.py"
if (assert_clean_release_python_source "$release" >/dev/null 2>&1); then
    exit 101
fi
rm -- "$source_root/sitecustomize.py"

printf '%s\n' 'native' > "$package/evaluation/shadow.so"
if (assert_clean_release_python_source "$release" >/dev/null 2>&1); then
    exit 102
fi
rm -- "$package/evaluation/shadow.so"

mkdir -- "$package/__pycache__"
if (assert_clean_release_python_source "$release" >/dev/null 2>&1); then
    exit 103
fi
rmdir -- "$package/__pycache__"

ln -- "$package/evaluation/gate.py" "$package/evaluation/hardlink.py"
if (assert_clean_release_python_source "$release" >/dev/null 2>&1); then
    exit 104
fi
rm -- "$package/evaluation/hardlink.py"
assert_clean_release_python_source "$release"
"""
        self._run_bash_function_contract(body)

        sync = (self.script_root / "Sync-DgxBrainRelease.ps1").read_text(
            encoding="utf-8"
        )
        for required in (
            "brain/src may contain only regular .py files below irene_brain",
            "py[co]|pyd|so|dll|dylib",
            "brain/src/irene_brain/__init__.py",
            "tar -tvzf",
            "archive contains a link or special entry",
            "-type f -links +1",
            'assert_clean_release_python_source "$staging"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, sync + self.remote)

    def test_release_sync_requires_one_canonical_fixed_registration(self) -> None:
        python_path = self._git_bash_path(Path(sys.executable))
        canonical = (
            '{"qualification_id":"rcq_v2_reference_v1","schema_version":2,'
            '"workspace_protocol":{"contract":"pseudo-brain-workspace-v2",'
            '"registration_release_relative_path":'
            '"registrations/rcq-v2-reference-v1.json"}}'
        )
        body = f"""
PATH=/usr/bin:/bin
python3() {{ "{python_path}" "$@"; }}
release="$1/release"
registration="$release/registrations/rcq-v2-reference-v1.json"
mkdir -p -- "$(dirname -- "$registration")"
printf '%s\n' '{canonical}' > "$registration"
assert_canonical_rcq_v2_registration_file "$registration" "$release"

printf '%s\n' '{{"schema_version":2, "qualification_id":"rcq_v2_reference_v1", "workspace_protocol":{{"contract":"pseudo-brain-workspace-v2","registration_release_relative_path":"registrations/rcq-v2-reference-v1.json"}}}}' > "$registration"
if (assert_canonical_rcq_v2_registration_file "$registration" "$release" >/dev/null 2>&1); then
    exit 101
fi

printf '%s\n' '{{"qualification_id":"rcq_v2_reference_v1","qualification_id":"rcq_v2_reference_v1","schema_version":2,"workspace_protocol":{{"contract":"pseudo-brain-workspace-v2","registration_release_relative_path":"registrations/rcq-v2-reference-v1.json"}}}}' > "$registration"
if (assert_canonical_rcq_v2_registration_file "$registration" "$release" >/dev/null 2>&1); then
    exit 102
fi
"""
        self._run_bash_function_contract(body)

        sync = (self.script_root / "Sync-DgxBrainRelease.ps1").read_text(
            encoding="utf-8"
        )
        for required in (
            "$sourceFiles -notcontains $registrationRelativePath",
            "build the fixed create-only",
            "registration_members == 1",
            "archive must contain the fixed RCQ-v2 registration exactly once",
            "assert_canonical_rcq_v2_registration_file",
            "fixed RCQ-v2 registration is not one canonical target-blind object",
        ):
            with self.subTest(required=required):
                self.assertIn(required, sync + self.remote)

    def test_pretraining_pin_is_create_only_chronology_and_image_bound(self) -> None:
        wrapper = (
            self.script_root / "New-DgxRcqV2PretrainingPin.ps1"
        ).read_text(encoding="utf-8")
        for required in (
            "$ReleaseId",
            "$ReleaseArchiveSha256",
            "$RegistrationSha256",
            "$ContainerImage",
            "$ContainerImageId",
            "Assert-DgxImageId",
            "'rcq_v2_pin_pretraining_v1'",
        ):
            with self.subTest(wrapper_required=required):
                self.assertIn(required, wrapper)
        for forbidden in ("RemoteWorkDir", "RunId", "ConfigRelativePath"):
            with self.subTest(wrapper_forbidden=forbidden):
                self.assertNotIn(forbidden, wrapper)

        for required in (
            "resolve_canonical_rcq_v2_workspace",
            "host_account_home_relative_path",
            "qualification-pins/rcq-v2-reference-v1",
            "pretraining.json",
            "PSEUDOBRAINRCQPRETRAINPIN\\x01",
            "marker_file_sha256",
            "container_image_id",
            "image_id_mismatch",
            "release_archive_hash_mismatch",
            "registration_hash_mismatch",
            "pretraining_chronology_violation",
            '"$workspace/runs/$RCQ_REFERENCE_RUN_ID"',
            '"$workspace/logs/${RCQ_REFERENCE_RUN_ID}.log"',
            '"$workspace/smoke-receipts/${release_id}.env"',
            '"$workspace/rcq-staging-canary-receipts/${release_id}.env"',
            "range_already_claimed",
            "os.O_EXCL | os.O_NOFOLLOW",
            "chmod 400",
            "pretraining pin must have mode 400",
            "pretraining_pin_sha256=",
        ):
            with self.subTest(remote_required=required):
                self.assertIn(required, self.remote)
        self.assertEqual(self.remote.count("rcq_v2_pin_pretraining_v1)"), 1)

    def test_preflight_checks_spark_specific_resources(self) -> None:
        for required in (
            "aarch64",
            "nvidia-smi",
            "MemAvailable",
            "system_python",
            "import platform, torch",
            "image_not_cached",
            "gpu_compute_processes=none",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.remote)

    def test_smoke_is_foreground_bounded_and_receipted(self) -> None:
        self.assertIn("timeout --signal=TERM --kill-after=15s 300s", self.remote)
        self.assertIn("PSEUDO_BRAIN_MAX_STEPS=3", self.remote)
        self.assertGreaterEqual(self.remote.count("CUBLAS_WORKSPACE_CONFIG=:4096:8"), 3)
        self.assertIn("loss.backward()", self.remote)
        self.assertIn("for test_path in tests/test_*.py", self.remote)
        self.assertIn(
            r"python3 -I -c '$UNITTEST_ISOLATED_BOOTSTRAP' \"\$test_module\"",
            self.remote,
        )
        self.assertIn("import sys,unittest;module=sys.argv[1]", self.remote)
        self.assertIn("unittest.main(module=None)", self.remote)
        self.assertNotIn('runpy.run_module("unittest"', self.remote)
        self.assertIn('/workspace/repo/brain");sys.argv=["unittest"', self.remote)
        self.assertIn("smoke-receipts", self.remote)
        self.assertIn("smoke_image_mismatch", self.remote)

    def test_schema3_staging_canary_is_fixed_two_phase_and_receipted(self) -> None:
        config_path = (
            self.brain_root
            / "configs"
            / "training"
            / "dgx-rcq-v2-staging-canary.toml"
        )
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], 3)
        self.assertEqual(config["run"]["name"], "dgx-rcq-v2-staging-canary")
        self.assertEqual(config["run"]["max_optimizer_steps"], 2)
        self.assertEqual(config["run"]["model_factory"].split(":")[-1], "build_smoke_model")
        self.assertEqual(config["dataset"]["seed_offset"], 0)
        self.assertEqual(config["dataset"]["test_sequences"], 1)
        self.assertEqual(
            [(stage["start_optimizer_step"], stage["end_optimizer_step"]) for stage in config["stages"]],
            [(0, 1), (1, 2)],
        )
        self.assertEqual(config["stages"][0]["trainable_parameters"], ["*"])
        self.assertEqual(
            config["stages"][1]["trainable_parameters"],
            ["model.value_per_thought.bias", "model.value_per_thought.weight"],
        )
        self.assertTrue(all(stage["reset_optimizer"] for stage in config["stages"]))
        self.assertTrue(
            all(
                stage["transition_gate"] == "none"
                and stage["completion_gate"] == "none"
                for stage in config["stages"]
            )
        )
        self.assertEqual(
            config["stages"][1]["invariance_audit"],
            "nonvalue_action_state_v1",
        )
        self.assertEqual(config["precision"]["device"], "cuda")
        self.assertEqual(config["precision"]["mode"], "bfloat16")
        self.assertEqual(config["logging"]["checkpoint_every_steps"], 1)

        launcher = (
            self.script_root / "Invoke-DgxRcqStagingCanary.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("'rcq_staging_canary'", launcher)
        self.assertNotIn("LaunchMode", launcher)
        self.assertNotIn("AcknowledgeDetached", launcher)
        for required in (
            "RCQ_STAGING_CANARY_CONFIG",
            "--stop-after-step 1",
            "step-00000001.pt",
            "checkpoint_schema=2",
            "stage-transition-01.json",
            'stage_state.get("trainable_parameter_names")',
            "invariance-step-00000002.json",
            "metrics do not prove exact prefix resume",
            "timeout --signal=TERM --kill-after=15s 600s",
            "rcq-staging-canary-receipts",
            "require_rcq_staging_canary_receipt",
            "dedicated_canary_required",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.remote)
        self.assertNotIn('stage_state.get("trainable_parameters")', self.remote)

    def test_generated_staging_canary_job_is_shell_valid_and_two_phase(self) -> None:
        image_id = "sha256:" + ("a" * 64)
        body = f"""
PATH=/usr/bin:/bin
workspace="$1/projects/pseudo-brain"
mkdir -p -- "$workspace/releases/release-1" "$workspace/runs" "$workspace/logs"
write_rcq_staging_canary_job "$workspace" release-1 repo/image:tag canary-1 2 8 {image_id} 1000 1000
launch="$workspace/runs/canary-1/launch.sh"
bash -n "$launch"
[[ "$(grep -c -- '^docker run --rm' "$launch")" == 2 ]]
grep -F -- '--stop-after-step 1' "$launch" >/dev/null
grep -F -- '--resume "/workspace/run/checkpoints/step-00000001.pt"' "$launch" >/dev/null
grep -F -- 'metric_keys != [' "$launch" >/dev/null
"""
        self._run_bash_function_contract(body)

    def test_reference_launch_requires_untampered_staging_canary_receipt(self) -> None:
        image_id = "sha256:" + ("a" * 64)
        pin_file_sha = "c" * 64
        pin_semantic_sha = "d" * 64
        body = f"""
PATH=/usr/bin:/bin
workspace="$1/projects/pseudo-brain"
release="$workspace/releases/release-1"
run="$workspace/runs/canary-1"
receipt_root="$workspace/rcq-staging-canary-receipts"
mkdir -p -- "$release/brain/configs/training" "$run/checkpoints" "$receipt_root"
printf '%s\\n' 'config' > "$release/$RCQ_STAGING_CANARY_CONFIG"
printf '%s\\n' 'checkpoint' > "$run/checkpoints/step-00000002.pt"
printf '%s\\n' 'metrics' > "$run/metrics.jsonl"
printf '%s\\n' 'transition' > "$run/stage-transition-01.json"
printf '%s\\n' 'invariance' > "$run/invariance-step-00000002.json"
config_sha="$(sha256sum "$release/$RCQ_STAGING_CANARY_CONFIG" | awk '{{print $1}}')"
checkpoint_sha="$(sha256sum "$run/checkpoints/step-00000002.pt" | awk '{{print $1}}')"
metrics_sha="$(sha256sum "$run/metrics.jsonl" | awk '{{print $1}}')"
transition_sha="$(sha256sum "$run/stage-transition-01.json" | awk '{{print $1}}')"
invariance_sha="$(sha256sum "$run/invariance-step-00000002.json" | awk '{{print $1}}')"
cat > "$receipt_root/release-1.env" <<EOF
workspace_contract=pseudo-brain-workspace-v2
canary_contract=rcq-schema3-two-phase-v1
release_id=release-1
container_image=repo/image:tag
container_image_id={image_id}
pretraining_pin_file_sha256={pin_file_sha}
pretraining_pin_sha256={pin_semantic_sha}
config_path=$RCQ_STAGING_CANARY_CONFIG
config_sha256=$config_sha
run_id=canary-1
final_checkpoint=step-00000002.pt
final_checkpoint_sha256=$checkpoint_sha
metrics_sha256=$metrics_sha
stage_transition_sha256=$transition_sha
final_invariance_sha256=$invariance_sha
completed_utc=20990101T000000Z
EOF
require_rcq_staging_canary_receipt "$workspace" "$release" release-1 repo/image:tag {image_id} {pin_file_sha} {pin_semantic_sha}
printf '%s\\n' 'tampered' >> "$run/metrics.jsonl"
if (require_rcq_staging_canary_receipt "$workspace" "$release" release-1 repo/image:tag {image_id} {pin_file_sha} {pin_semantic_sha} >/dev/null 2>&1); then
    exit 104
fi
"""
        self._run_bash_function_contract(body)

    def test_resume_preflight_refuses_terminal_failed_stage_gates(self) -> None:
        source_root = str(self.brain_root / "src")
        sys.path.insert(0, source_root)
        try:
            from irene_brain.evaluation.rcq_v2 import (
                evaluate_rcq_v2_development,
                evaluate_rcq_v2_value_development,
            )
            from irene_brain.training.protocol import TrainingStepResult
        finally:
            sys.path.remove(source_root)

        python_path = self._git_bash_path(Path(sys.executable))
        release_temporary = tempfile.TemporaryDirectory(
            prefix="dgx-current-gate-evaluator-"
        )
        self.addCleanup(release_temporary.cleanup)
        release_root = Path(release_temporary.name)
        shutil.copytree(
            self.brain_root / "src" / "irene_brain",
            release_root / "brain" / "src" / "irene_brain",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        release = self._git_bash_path(release_root)

        def metrics(*, value_loss: float, passing_action: bool) -> dict[str, float]:
            return {
                "movement_target_active_count": 2_275 / 1_536,
                "movement_changed_samples_per_sample": 467 / 1_536,
                "previous_control_movement_exact_match": 1_069 / 1_536,
                "movement_exact_match": (1_300 if passing_action else 0) / 1_536,
                "movement_changed_exact_matches_per_sample": (
                    (300 if passing_action else 0) / 1_536
                ),
                "positive_key_recall": 0.95 if passing_action else 0.0,
                "movement_false_positive_count": 0.0,
                "movement_opposite_conflict_rate": 0.0,
                "non_movement_key_false_positive_count": 0.0,
                "off_support_button_positive_count": 0.0,
                "off_support_button_target_active_count": 0.0,
                "continuous_target_nonzero_count": 0.0,
                "continuous_action_squared_magnitude": 0.0,
                "continuous_action_outside_0_05_count": 0.0,
                "total_loss": value_loss,
                "value_loss": value_loss,
            }

        failed_entry = evaluate_rcq_v2_development(
            TrainingStepResult(
                loss=0.0,
                metrics=metrics(value_loss=1.0, passing_action=False),
                samples=1_536,
            )
        )
        passed_entry = evaluate_rcq_v2_development(
            TrainingStepResult(
                loss=0.0,
                metrics=metrics(value_loss=1.0, passing_action=True),
                samples=1_536,
            )
        )
        self.assertFalse(failed_entry.passed)
        self.assertTrue(passed_entry.passed)
        passed_entry_encoded = (passed_entry.canonical_json + "\n").encode("utf-8")
        failed_completion = evaluate_rcq_v2_value_development(
            TrainingStepResult(
                loss=0.0,
                metrics=metrics(value_loss=1.0, passing_action=True),
                samples=1_536,
            ),
            entry_report=passed_entry,
            entry_report_sha256=sha256(passed_entry_encoded).hexdigest(),
        )
        passed_completion = evaluate_rcq_v2_value_development(
            TrainingStepResult(
                loss=0.0,
                metrics=metrics(value_loss=0.1, passing_action=True),
                samples=1_536,
            ),
            entry_report=passed_entry,
            entry_report_sha256=sha256(passed_entry_encoded).hexdigest(),
        )
        self.assertFalse(failed_completion.passed)
        self.assertTrue(passed_completion.passed)

        tampered_entry_gate = passed_completion.to_dict()
        tampered_entry_gate["entry_gate"]["report_sha256"] = "0" * 64
        tampered_entry_gate_json = json.dumps(
            tampered_entry_gate,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        tampered_baselines = passed_completion.to_dict()
        tampered_baselines["registered_value_baselines"][
            "absolute_max_mse_hex"
        ] = "0x0.0p+0"
        tampered_baselines_json = json.dumps(
            tampered_baselines,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        tampered_aggregate = failed_completion.to_dict()
        tampered_aggregate["passed"] = True
        tampered_aggregate_json = json.dumps(
            tampered_aggregate,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        body = f"""
PATH=/usr/bin:/bin
python3() {{ "{python_path}" "$@"; }}
run="$1/run"
mkdir -p -- "$run"
printf '%s\n' '{failed_entry.canonical_json}' > "$run/development-gate-step-00001536.json"
if (refuse_terminal_failed_stage_gate "$run" 1536 "{release}" > /dev/null 2> "$run/error"); then
    exit 101
fi
grep -F -- 'code=terminal_transition_gate_failed' "$run/error" > /dev/null
printf '%s\n' '{passed_entry.canonical_json}' > "$run/development-gate-step-00001536.json"
refuse_terminal_failed_stage_gate "$run" 1536 "{release}"
printf '%s\n' '{failed_completion.canonical_json}' > "$run/final-development-step-00002048.json"
if (refuse_terminal_failed_stage_gate "$run" 2048 "{release}" > /dev/null 2> "$run/error"); then
    exit 102
fi
grep -F -- 'code=terminal_completion_gate_failed' "$run/error" > /dev/null
printf '%s\n' '{passed_completion.canonical_json}' > "$run/final-development-step-00002048.json"
refuse_terminal_failed_stage_gate "$run" 2048 "{release}"
printf '%s\n' '{tampered_entry_gate_json}' > "$run/final-development-step-00002048.json"
if (refuse_terminal_failed_stage_gate "$run" 2048 "{release}" >/dev/null 2>&1); then
    exit 103
fi
printf '%s\n' '{tampered_baselines_json}' > "$run/final-development-step-00002048.json"
if (refuse_terminal_failed_stage_gate "$run" 2048 "{release}" >/dev/null 2>&1); then
    exit 104
fi
printf '%s\n' '{passed_entry.canonical_json}' > "$run/final-development-step-00002048.json"
if (refuse_terminal_failed_stage_gate "$run" 2048 "{release}" >/dev/null 2>&1); then
    exit 105
fi
printf '%s\n' '{tampered_aggregate_json}' > "$run/final-development-step-00002048.json"
if (refuse_terminal_failed_stage_gate "$run" 2048 "{release}" >/dev/null 2>&1); then
    exit 106
fi
rm -- "$run/development-gate-step-00001536.json"
printf '%s\n' '{passed_completion.canonical_json}' > "$run/final-development-step-00002048.json"
if (refuse_terminal_failed_stage_gate "$run" 2048 "{release}" >/dev/null 2>&1); then
    exit 107
fi
"""
        self._run_bash_function_contract(body)

        with tempfile.TemporaryDirectory(
            prefix="dgx-legacy-transition-evaluator-"
        ) as temporary:
            legacy_release = Path(temporary)
            legacy_package = legacy_release / "brain" / "src" / "irene_brain"
            shutil.copytree(
                self.brain_root / "src" / "irene_brain",
                legacy_package,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            legacy_evaluator = legacy_package / "evaluation" / "rcq_v2.py"
            source = legacy_evaluator.read_text(encoding="utf-8")
            old_definition = "def evaluate_rcq_v2_value_development("
            self.assertEqual(source.count(old_definition), 1)
            legacy_evaluator.write_text(
                source.replace(
                    old_definition,
                    "def legacy_removed_rcq_v2_value_development(",
                ),
                encoding="utf-8",
                newline="\n",
            )
            legacy_release_bash = self._git_bash_path(legacy_release)
            transition_only = f"""
PATH=/usr/bin:/bin
python3() {{ "{python_path}" "$@"; }}
run="$1/legacy-transition-run"
mkdir -p -- "$run"
printf '%s\n' '{passed_entry.canonical_json}' > "$run/development-gate-step-00001536.json"
refuse_terminal_failed_stage_gate "$run" 1536 "{legacy_release_bash}"
"""
            self._run_bash_function_contract(transition_only)

        for required in (
            'refuse_terminal_failed_stage_gate "$run_dir" "$checkpoint_step" "$release"',
            "rcq_v2_development_v1",
            "evaluate_rcq_v2_development",
            "evaluate_rcq_v2_value_development",
            "registered_value_baselines",
            "entry_report_sha256",
            "terminal_transition_gate_failed",
            "terminal_completion_gate_failed",
            "invalid_terminal_gate_artifact",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.remote)
        refusal = self.remote.index(
            'refuse_terminal_failed_stage_gate "$run_dir" "$checkpoint_step" "$release"'
        )
        completion_fallback = self.remote.index(
            "fail training_complete 'selected checkpoint already reached",
            refusal,
        )
        attempt_claim = self.remote.index(
            'mkdir -p -- "$run_dir/resume-attempts"',
            refusal,
        )
        self.assertLess(refusal, completion_fallback)
        self.assertLess(refusal, attempt_claim)

    def test_detached_training_requires_explicit_acknowledgement(self) -> None:
        launcher = (self.script_root / "Start-DgxBrainTraining.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("[ValidateSet('Foreground', 'Tmux')]", launcher)
        self.assertIn("$AcknowledgeDetached", launcher)
        self.assertIn("No persistent job is started implicitly", launcher)

    def test_resume_requires_latest_pointer_or_exact_checkpoint_digest(self) -> None:
        launcher = (self.script_root / "Resume-DgxBrainTraining.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("DefaultParameterSetName = 'Latest'", launcher)
        self.assertIn("ParameterSetName = 'Latest'", launcher)
        self.assertIn("ParameterSetName = 'Exact'", launcher)
        self.assertIn("$UseLatest", launcher)
        self.assertIn("$CheckpointSha256", launcher)
        self.assertIn("Assert-DgxSha256", launcher)
        self.assertIn(r"^step-[0-9]{8}\.pt$", launcher)
        self.assertIn("Tmux resume requires -AcknowledgeDetached", launcher)

    def test_resume_preserves_identity_and_refuses_active_or_older_state(self) -> None:
        for required in (
            "verify_original_run_boundaries",
            "workspace_contract=pseudo-brain-workspace-v2",
            "runtime_boundary_mismatch",
            "container_image_id",
            "container_cpu_count",
            "container_memory_gib",
            "min_free_disk_gib",
            "min_available_memory_gib",
            "container_uid",
            "container_gid",
            "assert_run_inactive",
            "assert_run_lock_available",
            "wait_for_detached_launch",
            'docker ps -a --filter "name=^/pseudo-brain-${run_id}$"',
            "--entrypoint /usr/bin/python3",
            "TRAIN_ISOLATED_BOOTSTRAP",
            "-I -c '$TRAIN_ISOLATED_BOOTSTRAP'",
            "-e PSEUDO_BRAIN_RUN_DIR=/workspace/run",
            "-e TOKENIZERS_PARALLELISM=false",
            "checkpoint_not_highest",
            "training_complete",
            "checkpoint_hash_mismatch",
            "latest.json failed strict validation",
            "--resume \"/workspace/run/checkpoints/$checkpoint_name\"",
            '--checkpoint-sha256 "$checkpoint_sha"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.remote)
        self.assertNotIn("if grep -q '^launch_sha256='", self.remote)
        self.assertIn(
            "grep -F -- '-e PYTHONPATH=' \"$launch\" >/dev/null &&",
            self.remote,
        )
        self.assertIn(
            "-e BASH_ENV= -e ENV= -e PYTHONHOME= -e PYTHONPATH= \\",
            self.remote,
        )
        self.assertNotRegex(self.remote, r"-e PYTHONPATH=/")
        self.assertNotIn("-e IRENE_BRAIN_", self.remote)
        self.assertIn('metadata_value "$metadata" launch_sha256', self.remote)
        self.assertIn('"$image_id" -I -c', self.remote)
        self.assertIn(
            'require_file_contains "$launch" "\\"$current_image_id\\" -I -c"',
            self.remote,
        )

    def test_resume_attempts_are_append_only_and_separately_retrievable(self) -> None:
        retrieval = (self.script_root / "Receive-DgxBrainArtifact.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("resume-attempts", self.remote)
        self.assertIn("set -o noclobber", self.remote)
        self.assertIn('tee -a "$log_path"', self.remote)
        self.assertIn("training lock is already held", self.remote)
        self.assertIn("metrics_pre_resume_bytes", self.remote)
        self.assertIn("metrics_pre_resume_lines", self.remote)
        self.assertIn("resume_boundary checkpoint_step=", self.remote)
        self.assertIn("'ResumeAttempts' { 'resume-attempts' }", retrieval)

    def test_launch_writers_are_not_called_in_command_substitutions(self) -> None:
        self.assertNotRegex(
            self.remote,
            r'job_path="\$\([\s\\]*write_(?:training|resume)_job',
        )
        self.assertIn("write_training_job \\", self.remote)
        self.assertIn("write_resume_job \\", self.remote)
        self.assertIn("exec 8> \"$job_path\" || exit 75", self.remote)
        self.assertIn("set -o noclobber", self.remote)

    def test_same_run_and_attempt_collisions_preserve_all_artifacts(self) -> None:
        bash = self._bash_path()
        function_source = self.remote.split('action="${1:-}"', maxsplit=1)[0]
        image_id = "sha256:" + ("a" * 64)
        checkpoint_sha = "b" * 64
        body = f"""
PATH=/usr/bin:/bin
workspace="$1/projects/pseudo-brain"
mkdir -p -- "$workspace/releases/release-1" "$workspace/runs" "$workspace/logs"
write_training_job "$workspace" release-1 repo/image:tag brain/configs/training/dgx-smoke.toml run-1 2 8 {image_id} 1000 1000
printf '%s\n' 'original-run-env' > "$workspace/runs/run-1/run.env"
printf '%s\n' 'original-run-log' > "$workspace/logs/run-1.log"
training_launch_sha="$(sha256sum "$workspace/runs/run-1/launch.sh" | awk '{{print $1}}')"
if (write_training_job "$workspace" release-1 repo/image:tag brain/configs/training/dgx-smoke.toml run-1 2 8 {image_id} 1000 1000 >/dev/null 2>&1); then
    exit 91
fi
[[ "$(sha256sum "$workspace/runs/run-1/launch.sh" | awk '{{print $1}}')" == "$training_launch_sha" ]]
[[ "$(cat "$workspace/runs/run-1/run.env")" == original-run-env ]]
[[ "$(cat "$workspace/logs/run-1.log")" == original-run-log ]]

mkdir -p -- "$workspace/runs/run-1/resume-attempts"
attempt_id=resume-20990101T000000Z-bbbbbbbbbbbb
write_resume_job "$workspace" release-1 repo/image:tag brain/configs/training/dgx-smoke.toml run-1 2 8 step-00000042.pt {checkpoint_sha} "$attempt_id" {image_id} 123 4 1000 1000
attempt_dir="$workspace/runs/run-1/resume-attempts/$attempt_id"
printf '%s\n' 'original-resume-env' > "$attempt_dir/resume.env"
printf '%s\n' 'original-resume-log' > "$attempt_dir/train.log"
resume_launch_sha="$(sha256sum "$attempt_dir/launch.sh" | awk '{{print $1}}')"
if (write_resume_job "$workspace" release-1 repo/image:tag brain/configs/training/dgx-smoke.toml run-1 2 8 step-00000042.pt {checkpoint_sha} "$attempt_id" {image_id} 123 4 1000 1000 >/dev/null 2>&1); then
    exit 92
fi
[[ "$(sha256sum "$attempt_dir/launch.sh" | awk '{{print $1}}')" == "$resume_launch_sha" ]]
[[ "$(cat "$attempt_dir/resume.env")" == original-resume-env ]]
[[ "$(cat "$attempt_dir/train.log")" == original-resume-log ]]
"""
        with tempfile.TemporaryDirectory(prefix="dgx-collision-") as temporary:
            temporary_path = Path(temporary)
            test_script = temporary_path / "collision-contract.sh"
            test_script.write_text(function_source + body, encoding="utf-8", newline="\n")
            completed = subprocess.run(
                [
                    bash,
                    test_script.name,
                    ".",
                ],
                check=False,
                capture_output=True,
                cwd=temporary_path,
                text=True,
                timeout=15,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_detached_handshake_requires_live_tmux_and_exact_container(self) -> None:
        body = """
PATH=/usr/bin:/bin
ready="$1/launch.ready"
printf '%s\n' 'lock_acquired=1' > "$ready"
tmux() { return 1; }
docker() { printf '%s\n' pseudo-brain-run-1; }
if (wait_for_detached_launch pseudo-brain-run-1 "$ready" run-1 >/dev/null 2>&1); then
    exit 93
fi
tmux() { return 0; }
docker() { printf '%s\n' pseudo-brain-run-1; }
wait_for_detached_launch pseudo-brain-run-1 "$ready" run-1 >/dev/null
"""
        self._run_bash_function_contract(body)

    def test_canonical_path_guard_rejects_workspace_escape(self) -> None:
        body = """
PATH=/usr/bin:/bin
mkdir -p -- "$1/owned" "$1/outside"
if (assert_safe_directory_path "$1/outside" "$1/owned" escaped >/dev/null 2>&1); then
    exit 94
fi
assert_safe_directory_path "$1/owned" "$1/owned" owned >/dev/null
"""
        self._run_bash_function_contract(body)

    def test_original_run_identity_mismatches_fail_behaviorally(self) -> None:
        image_id = "sha256:" + ("a" * 64)
        body = f"""
PATH=/usr/bin:/bin
workspace="$1/projects/pseudo-brain"
config=brain/configs/training/test.toml
mkdir -p -- "$workspace/releases/release-1/brain/configs/training" \
    "$workspace/releases/release-1/brain/src/irene_brain/training" \
    "$workspace/runs" "$workspace/logs" "$workspace/datasets"
printf '%s\n' 'config' > "$workspace/releases/release-1/$config"
printf '%s\n' 'PACKAGE = 1' > "$workspace/releases/release-1/brain/src/irene_brain/__init__.py"
printf '%s\n' 'TRAINING = 1' > "$workspace/releases/release-1/brain/src/irene_brain/training/__init__.py"
printf '%s\n' 'trainer' > "$workspace/releases/release-1/brain/src/irene_brain/training/train.py"
write_training_job "$workspace" release-1 repo/image:tag "$config" run-1 2 8 {image_id} 1000 1000
canonical_workspace="$(realpath -e -- "$workspace")"
sed -i "s#${{workspace}}#${{canonical_workspace}}#g" "$workspace/runs/run-1/launch.sh"
printf '%s\n' '{{}}' > "$workspace/runs/run-1/resolved-config.json"
config_sha="$(sha256sum "$workspace/releases/release-1/$config" | awk '{{print $1}}')"
launch_sha="$(sha256sum "$workspace/runs/run-1/launch.sh" | awk '{{print $1}}')"
cat > "$workspace/runs/run-1/run.env" <<EOF
workspace_contract=pseudo-brain-workspace-v2
release_id=release-1
container_image=repo/image:tag
container_image_id={image_id}
config_path=$config
config_sha256=$config_sha
min_free_disk_gib=1
min_available_memory_gib=4
container_cpu_count=2
container_memory_gib=8
container_uid=1000
container_gid=1000
network=none
ipc=host
pids_limit=2048
cublas_workspace_config=:4096:8
launch_sha256=$launch_sha
EOF
verify_original_run_boundaries "$workspace" release-1 repo/image:tag "$config" run-1 2 8 {image_id} 1 4 1000 1000
if (verify_original_run_boundaries "$workspace" release-1 repo/image:tag "$config" run-1 2 8 {image_id} 0 4 1000 1000 >/dev/null 2>&1); then
    exit 95
fi
if (verify_original_run_boundaries "$workspace" release-1 repo/image:tag "$config" run-1 2 8 {image_id} 1 4 1001 1000 >/dev/null 2>&1); then
    exit 96
fi
sed -i 's/container_image_id=sha256:a/container_image_id=sha256:c/' "$workspace/runs/run-1/run.env"
if (verify_original_run_boundaries "$workspace" release-1 repo/image:tag "$config" run-1 2 8 {image_id} 1 4 1000 1000 >/dev/null 2>&1); then
    exit 97
fi
"""
        self._run_bash_function_contract(body)

    def test_status_and_artifact_retrieval_reject_links_and_use_canonical_paths(self) -> None:
        retrieval = (self.script_root / "Receive-DgxBrainArtifact.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("assert_safe_regular_file", self.remote)
        self.assertIn("assert_safe_artifact_tree", self.remote)
        self.assertIn(r"\! -type d \! -type f", self.remote)
        self.assertIn("artifact_path=%s", self.remote)
        self.assertIn("$canonicalRemotePath", retrieval)
        self.assertIn("one canonical, absolute artifact path", retrieval)

    def test_local_artifact_destination_is_anchored_to_pseudo_brain(self) -> None:
        retrieval_path = self.script_root / "Receive-DgxBrainArtifact.ps1"
        retrieval = retrieval_path.read_text(encoding="utf-8")
        for required in (
            "Join-Path $PSScriptRoot '..\\..\\..'",
            "[IO.Path]::IsPathRooted",
            "must remain inside the Pseudo-Brain workspace",
            "ReparsePoint",
        ):
            with self.subTest(required=required):
                self.assertIn(required, retrieval)

        powershell = self._powershell_path()
        with tempfile.TemporaryDirectory(prefix="dgx-outside-destination-") as temporary:
            destination = Path(temporary) / "retrieved"
            completed = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(retrieval_path),
                    "-SshTarget",
                    "must-not-be-resolved",
                    "-RemoteWorkDir",
                    "~/projects/pseudo-brain",
                    "-RunId",
                    "run-1",
                    "-Kind",
                    "Logs",
                    "-LocalDestination",
                    str(destination),
                    "-PlanOnly",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must remain inside the Pseudo-Brain workspace", completed.stderr)
        self.assertNotIn("OpenSSH", completed.stderr)

    @staticmethod
    def _bash_path() -> str:
        if os.name == "nt":
            program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            candidate = program_files / "Git" / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
        candidate = shutil.which("bash")
        if candidate is None:
            raise unittest.SkipTest("Bash is unavailable for the collision contract")
        return candidate

    @staticmethod
    def _powershell_path() -> str:
        candidate = shutil.which("powershell") or shutil.which("pwsh")
        if candidate is not None:
            return candidate
        if os.name == "nt":
            system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            windows_powershell = (
                system_root
                / "System32"
                / "WindowsPowerShell"
                / "v1.0"
                / "powershell.exe"
            )
            if windows_powershell.is_file():
                return str(windows_powershell)
        raise unittest.SkipTest("PowerShell is unavailable for the local path contract")

    @staticmethod
    def _git_bash_path(path: Path) -> str:
        resolved = path.resolve()
        if os.name != "nt":
            return resolved.as_posix()
        drive = resolved.drive.rstrip(":").lower()
        tail = resolved.as_posix()[len(resolved.drive) :].lstrip("/")
        return f"/{drive}/{tail}"

    def _run_bash_function_contract(self, body: str) -> None:
        bash = self._bash_path()
        function_source = self.remote.split('action="${1:-}"', maxsplit=1)[0]
        with tempfile.TemporaryDirectory(prefix="dgx-function-contract-") as temporary:
            temporary_path = Path(temporary)
            test_script = temporary_path / "function-contract.sh"
            test_script.write_text(function_source + body, encoding="utf-8", newline="\n")
            completed = subprocess.run(
                [bash, test_script.name, "."],
                check=False,
                capture_output=True,
                cwd=temporary_path,
                text=True,
                timeout=15,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_sync_filters_secrets_artifacts_and_weights(self) -> None:
        sync = (self.script_root / "Sync-DgxBrainRelease.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("Git\\cmd\\git.exe", self.common)
        self.assertIn('chmod -R a-w -- "$staging"', self.remote)
        self.assertIn('"pseudo-brain-$nonce.tgz"', sync)
        self.assertIn(r"^pseudo-brain-[a-f0-9]{12}\.tgz$", self.remote)
        self.assertIn(".pseudo-brain-workspace-v2", sync)
        self.assertIn("rev-parse --show-toplevel", sync)
        self.assertIn("canonical Pseudo-Brain Git root", sync)
        self.assertIn("registrations/rcq-v2-reference-v1.json", sync)
        for forbidden_pattern in ("\\.env", "checkpoints?", "safetensors", "ReparsePoint"):
            with self.subTest(forbidden_pattern=forbidden_pattern):
                self.assertIn(forbidden_pattern, sync)

    def test_production_rcq_actions_bind_pins_and_reject_identity_substitution(self) -> None:
        pin_derived = {
            "Invoke-DgxRcqV2Smoke.ps1": "rcq_v2_smoke_v1",
            "Invoke-DgxRcqStagingCanary.ps1": "rcq_staging_canary",
            "Start-DgxRcqV2Reference.ps1": "rcq_v2_reference_train_v1",
            "Resume-DgxRcqV2Reference.ps1": "rcq_v2_reference_resume_v1",
            "Invoke-DgxRcqV2Preclaim.ps1": "rcq_v2_preclaim_v1",
            "Invoke-DgxRcqV2FinalOnce.ps1": "rcq_v2_final_once_v1",
            "Test-DgxRcqV2FinalReceipt.ps1": "rcq_v2_verify_receipt_v1",
        }
        for filename, action in pin_derived.items():
            wrapper = (self.script_root / filename).read_text(encoding="utf-8")
            with self.subTest(wrapper=filename):
                self.assertIn(f"'{action}'", wrapper)
                self.assertNotIn("RemoteWorkDir", wrapper)
                self.assertNotIn("ReleaseId", wrapper)
                self.assertNotIn("ContainerImage", wrapper)
                self.assertNotIn("ConfigRelativePath", wrapper)
                self.assertNotIn("RunId", wrapper)
                self.assertNotIn("$env:", wrapper)
        authorization = (
            self.script_root / "New-DgxRcqV2FinalAuthorization.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("'rcq_v2_authorize_final_v1'", authorization)
        self.assertNotIn("RemoteWorkDir", authorization)
        self.assertNotIn("ReleaseId", authorization)
        self.assertNotIn("ContainerImage", authorization)
        self.assertNotIn("ConfigRelativePath", authorization)
        self.assertNotIn("RunId", authorization)
        reference_start = (
            self.script_root / "Start-DgxRcqV2Reference.ps1"
        ).read_text(encoding="utf-8")
        reference_resume = (
            self.script_root / "Resume-DgxRcqV2Reference.ps1"
        ).read_text(encoding="utf-8")
        for wrapper in (reference_start, reference_resume):
            self.assertNotIn("ContainerCpuCount", wrapper)
            self.assertNotIn("ContainerMemoryGiB", wrapper)
            self.assertNotIn("MinFreeDiskGiB", wrapper)

        for required in (
            "rcq_v2_preclaim_v1)",
            "rcq_v2_final_once_v1)",
            "rcq_v2_verify_receipt_v1)",
            "rcq_v2_authorize_final_v1)",
            "rcq_v2_reference_train_v1)",
            "rcq_v2_reference_resume_v1)",
            "accepts no caller-selected identity or runtime arguments",
            "bind_rcq_v2_pretraining_authority",
            "bind_rcq_v2_receipt_verification_authority",
            "open_rcq_v2_range_authority_lock",
            "prepare_rcq_v2_evaluator_paths",
            "run_rcq_v2_evaluator_container",
            "write_rcq_v2_final_authorization",
            "cpu_limit=12",
            "memory_limit_gib=96",
            "--pull never",
            "--network none",
            "-v \"$RCQ_PIN_DIRECTORY:/workspace/pins:ro\"",
            "dedicated_reference_required",
            "the RCQ-v2 reference config/run requires Start-DgxRcqV2Reference.ps1",
            "the RCQ-v2 reference config/run requires Resume-DgxRcqV2Reference.ps1",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.remote)
        self.assertNotIn("${RELEASE_ID", self.remote)
        self.assertNotIn("${RUN_ID", self.remote)
        self.assertNotIn("$DGX_RELEASE_ID", self.remote)
        self.assertNotIn("PSEUDO_BRAIN_RELEASE", self.remote)
        generic_train = (self.script_root / "Start-DgxBrainTraining.ps1").read_text(
            encoding="utf-8"
        )
        generic_resume = (self.script_root / "Resume-DgxBrainTraining.ps1").read_text(
            encoding="utf-8"
        )
        generic_smoke = (self.script_root / "Invoke-DgxBrainSmoke.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("requires Start-DgxRcqV2Reference.ps1", generic_train)
        self.assertIn("requires Resume-DgxRcqV2Reference.ps1", generic_resume)
        self.assertIn("requires Invoke-DgxRcqV2Smoke.ps1", generic_smoke)
        self.assertIn("must use Invoke-DgxRcqStagingCanary.ps1", generic_train)
        self.assertIn("must use Invoke-DgxRcqStagingCanary.ps1", generic_resume)
        self.assertGreaterEqual(self.remote.count("dedicated_canary_required"), 2)

    def test_canonical_ssh_and_scripts_ignore_path_and_parse(self) -> None:
        powershell = self._powershell_path()
        bash = self._bash_path()
        syntax = subprocess.run(
            [bash, "-n", str(self.script_root / "_remote_dispatch.sh")],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(syntax.returncode, 0, msg=syntax.stderr)

        with tempfile.TemporaryDirectory(prefix="dgx-ssh-shadow-") as temporary:
            shadow = Path(temporary)
            fake_ssh = shadow / "ssh.cmd"
            fake_ssh.write_text("@echo FAKE-SSH>&2\r\n@exit 99\r\n", encoding="utf-8")
            probe = shadow / "probe.ps1"
            probe.write_text(
                "\n".join(
                    (
                        "$ErrorActionPreference = 'Stop'",
                        f". '{self.script_root / 'Dgx.Common.ps1'}'",
                        "$ssh = Get-DgxApplication -Name ssh",
                        "Write-Output $ssh",
                    )
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            environment = dict(os.environ)
            environment["PATH"] = str(shadow) + os.pathsep + environment.get("PATH", "")
            completed = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(probe),
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
                timeout=15,
            )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        resolved = completed.stdout.strip()
        self.assertTrue(resolved.lower().endswith(r"system32\openssh\ssh.exe"), resolved)
        self.assertNotIn("FAKE-SSH", completed.stderr)

        parser = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    "$failed = @(); "
                    f"Get-ChildItem -LiteralPath '{self.script_root}' -Filter '*.ps1' | "
                    "ForEach-Object { "
                    "$tokens = $null; $errors = $null; "
                    "[void][System.Management.Automation.Language.Parser]::ParseFile("
                    "$_.FullName, [ref]$tokens, [ref]$errors); "
                    "if ($errors.Count -gt 0) { $failed += ($_.Name + ': ' + $errors[0].Message) } "
                    "}; "
                    "if ($failed.Count -gt 0) { throw ($failed -join '; ') }"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(parser.returncode, 0, msg=parser.stderr or parser.stdout)

    def test_local_registration_bootstrap_survives_powershell_quoting(self) -> None:
        script = (self.script_root / "New-RcqV2Registration.ps1").read_text(
            encoding="utf-8"
        )
        helper = self.brain_root / "scripts" / "run_rcq_v2_registration.py"
        self.assertTrue(helper.is_file())
        helper_text = helper.read_text(encoding="utf-8")
        self.assertIn("runpy.run_module", helper_text)
        self.assertIn("irene_brain.evaluation.rcq_v2_registration", helper_text)
        self.assertIn("run_rcq_v2_registration.py", script)
        self.assertIn("-I -B", script)
        self.assertNotIn("-c $bootstrap", script)
        self.assertNotIn('python -I -c', script)

    def test_isolated_python_bootstrap_ignores_startup_hooks(self) -> None:
        python_path = self._git_bash_path(Path(sys.executable))
        body = f"""
PATH=/usr/bin:/bin
python3() {{ "{python_path}" "$@"; }}
printf '%s\\n' 'raise SystemExit(99)' > "$1/startup.py"
printf '%s\\n' 'raise SystemExit(98)' > "$1/sitecustomize.py"
export PYTHONSTARTUP="$1/startup.py"
export PYTHONPATH="$1"
export PYTHONHOME="$1"
output="$(python3 -I -c 'print(1)')"
[[ "$output" == 1 ]]
"""
        self._run_bash_function_contract(body)

    @staticmethod
    def _tree_identity(root: Path) -> tuple[tuple[str, str, bytes], ...]:
        identity: list[tuple[str, str, bytes]] = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                identity.append((relative, "directory", b""))
            elif path.is_file():
                identity.append((relative, "file", path.read_bytes()))
            else:
                identity.append((relative, "special", b""))
        return tuple(identity)


if __name__ == "__main__":
    unittest.main()
