#!/bin/bash -p
PATH='/usr/sbin:/usr/bin'
export PATH
unset BASH_ENV ENV PYTHONHOME PYTHONPATH CDPATH
LC_ALL=C
LANG=C
TZ=UTC
export LC_ALL LANG TZ
DGX_ACCOUNT_UID="$(/usr/bin/id -u)" || exit 97
if [[ "$(/usr/bin/uname -s)" == Linux ]]; then
    DGX_ACCOUNT_HOME="$(/usr/bin/getent passwd "$DGX_ACCOUNT_UID" | /usr/bin/awk -F: 'NR == 1 {print $6}')" || exit 97
else
    DGX_ACCOUNT_HOME="${HOME:-}"
fi
[[ "$DGX_ACCOUNT_HOME" == /* && "$DGX_ACCOUNT_HOME" != *$'\n'* ]] || exit 97
HOME="$DGX_ACCOUNT_HOME"
export HOME
set -euo pipefail

RCQ_STAGING_CANARY_CONFIG='brain/configs/training/dgx-rcq-v2-staging-canary.toml'
RCQ_REFERENCE_CONFIG='brain/configs/training/dgx-rcq-v2-reference.toml'
DGX_SMOKE_CONFIG='brain/configs/training/dgx-smoke.toml'
RCQ_REGISTRATION_RELATIVE_PATH='registrations/rcq-v2-reference-v2.json'
RCQ_REFERENCE_RUN_ID='dgx-rcq-v2-reference-seed-1702'
RCQ_PIN_DIRECTORY_RELATIVE_PATH='qualification-pins/rcq-v2-reference-v2'
RCQ_PRETRAINING_PIN_FILENAME='pretraining.json'
RCQ_FINAL_AUTHORIZATION_FILENAME='final-authorization.json'
RCQ_READINESS_RELATIVE_PATH='preclaim-readiness/rcq-v2-reference-v2.json'
RCQ_ENTRY_CHECKPOINT_NAME='step-00001536.pt'
RCQ_FINAL_CHECKPOINT_NAME='step-00002048.pt'
RCQ_WORKSPACE_MARKER_SHA256='5657bd63ee6ff61d1d90ea63b4cb02f9dd7d5dc3df223c9cbb028f9e2a096053'
TRAIN_ISOLATED_BOOTSTRAP='import runpy,sys;sys.path.insert(0,"/workspace/repo/brain/src");sys.argv[0]="irene_brain.training.train";runpy.run_module("irene_brain.training.train",run_name="__main__")'
UNITTEST_ISOLATED_BOOTSTRAP='import sys,unittest;module=sys.argv[1];assert module.startswith("tests.test_") and module[11:].replace("_","").isalnum();sys.path.insert(0,"/workspace/repo/brain/src");sys.path.insert(0,"/workspace/repo/brain");sys.argv=["unittest","-v",module];unittest.main(module=None)'
RCQ_PRECLAIM_ISOLATED_BOOTSTRAP='import runpy,sys;sys.path.insert(0,"/workspace/repo/brain/src");sys.argv=["irene_brain.evaluation.rcq_v2_torch","preclaim"];runpy.run_module("irene_brain.evaluation.rcq_v2_torch",run_name="__main__")'
RCQ_FINAL_ONCE_ISOLATED_BOOTSTRAP='import runpy,sys;sys.path.insert(0,"/workspace/repo/brain/src");sys.argv=["irene_brain.evaluation.rcq_v2_torch","final-once"];runpy.run_module("irene_brain.evaluation.rcq_v2_torch",run_name="__main__")'
RCQ_VERIFY_RECEIPT_ISOLATED_BOOTSTRAP='import runpy,sys;sys.path.insert(0,"/workspace/repo/brain/src");sys.argv=["irene_brain.evaluation.rcq_v2_torch","verify-receipt"];runpy.run_module("irene_brain.evaluation.rcq_v2_torch",run_name="__main__")'

fail() {
    local code="$1"
    shift
    printf 'DGX_FAIL code=%s message=%s\n' "$code" "$*" >&2
    exit 1
}

require_command() {
    local command_path resolved owner mode
    command_path="$(type -P -- "$1")" || fail missing_command "required command '$1' is unavailable"
    [[ "$command_path" == /* ]] || fail unsafe_command "required command '$1' did not resolve to an absolute system path"
    resolved="$(/usr/bin/realpath -e -- "$command_path")" || fail unsafe_command "required command '$1' cannot be resolved"
    [[ -f "$resolved" && ! -L "$resolved" ]] || fail unsafe_command "required command '$1' is not a regular executable"
    read -r owner mode < <(/usr/bin/stat -Lc '%u %a' -- "$resolved") ||
        fail unsafe_command "required command '$1' ownership cannot be inspected"
    [[ "$owner" == 0 ]] || fail unsafe_command "required command '$1' is not root-owned"
    (( (8#$mode & 8#022) == 0 )) || fail unsafe_command "required command '$1' is group/other writable"
    [[ -x "$resolved" ]] || fail unsafe_command "required command '$1' is not executable"
}

assert_trusted_host_runtime() {
    [[ "$(/usr/bin/uname -s)" == Linux ]] || return 0
    local directory owner mode command_name bash_path
    for directory in /usr/sbin /usr/bin; do
        [[ -d "$directory" && ! -L "$directory" ]] || fail unsafe_system_path "trusted PATH component '$directory' is unsafe"
        read -r owner mode < <(/usr/bin/stat -Lc '%u %a' -- "$directory") ||
            fail unsafe_system_path "trusted PATH component '$directory' cannot be inspected"
        [[ "$owner" == 0 ]] || fail unsafe_system_path "trusted PATH component '$directory' is not root-owned"
        (( (8#$mode & 8#022) == 0 )) || fail unsafe_system_path "trusted PATH component '$directory' is group/other writable"
    done
    for command_name in \
        awk basename bash cat chmod date df dirname find getent grep head id ln mkdir mv \
        python3 realpath rm sed sha256sum sort stat tail tar tee timeout tr uname wc; do
        require_command "$command_name"
    done
    bash_path="$(type -P bash)"
    [[ "$(/usr/bin/realpath -e -- /bin/bash)" == "$(/usr/bin/realpath -e -- "$bash_path")" ]] ||
        fail unsafe_command '/bin/bash is not the trusted system Bash'
}

require_uint() {
    [[ "$2" =~ ^[0-9]+$ ]] || fail invalid_number "$1 must be an unsigned integer"
}

require_sha256() {
    [[ "$2" =~ ^[a-f0-9]{64}$ ]] || fail invalid_sha256 "$1 must be a lowercase SHA-256"
}

require_image_id() {
    [[ "$1" =~ ^sha256:[a-f0-9]{64}$ ]] || fail invalid_image_id 'Docker returned an invalid immutable image ID'
}

require_slug() {
    [[ "$2" =~ ^[a-z0-9][a-z0-9._-]{0,62}$ ]] || fail invalid_slug "$1 is not a safe 1-63 character slug"
}

require_relative_path() {
    local label="$1"
    local value="$2"
    [[ "$value" =~ ^brain/[A-Za-z0-9._/-]+$ ]] || fail invalid_path "$label must remain under brain/"
    [[ "/$value/" != *"/../"* && "$value" != *"//"* ]] || fail invalid_path "$label contains traversal"
}

require_image_reference() {
    [[ "$1" =~ ^[a-z0-9]+([._-][a-z0-9]+)*(/[a-z0-9]+([._-][a-z0-9]+)*)*(:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}|@sha256:[a-f0-9]{64})$ ]] ||
        fail invalid_image 'container image reference is not shell-safe'
}

resolve_workspace() {
    local raw="$1"
    [[ "$raw" =~ ^(~/|/)[A-Za-z0-9._/-]+$ ]] || fail invalid_workspace 'workspace must be an explicit POSIX path'
    [[ "/$raw/" != *"/../"* && "$raw" != *"//"* ]] || fail invalid_workspace 'workspace contains traversal'

    if [[ "$raw" == "~/"* ]]; then
        raw="$HOME/${raw#\~/}"
    fi
    local resolved
    resolved="$(realpath -m -- "$raw")"
    local leaf
    leaf="$(basename -- "$resolved")"
    [[ "$leaf" =~ ^pseudo-brain(-[A-Za-z0-9._-]+)?$ ]] ||
        fail unsafe_workspace "new workspace '$resolved' is not dedicated to pseudo-brain v2"
    [[ "$(basename -- "$(dirname -- "$resolved")")" == projects ]] ||
        fail unsafe_workspace 'new workspace must be a direct child of the remote projects directory'
    [[ "/$resolved/" != *"/irene-brain/"* ]] ||
        fail unsafe_workspace 'new workspace cannot be nested below the historical Irene workspace'
    [[ "$resolved" != / && "$resolved" != "$HOME" ]] || fail unsafe_workspace 'workspace is too broad'
    printf '%s\n' "$resolved"
}

resolve_historical_workspace() {
    local raw="$1"
    [[ "$raw" =~ ^(~/|/)[A-Za-z0-9._/-]+$ ]] || fail invalid_workspace 'historical workspace must be an explicit POSIX path'
    [[ "/$raw/" != *"/../"* && "$raw" != *"//"* ]] || fail invalid_workspace 'historical workspace contains traversal'

    if [[ "$raw" == "~/"* ]]; then
        raw="$HOME/${raw#\~/}"
    fi
    local resolved
    resolved="$(realpath -m -- "$raw")"
    [[ "$(basename -- "$resolved")" == irene-brain ]] ||
        fail unsafe_workspace "historical reads require the exact irene-brain workspace leaf"
    [[ "$(basename -- "$(dirname -- "$resolved")")" == projects ]] ||
        fail unsafe_workspace 'historical workspace must be a direct child of the remote projects directory'
    [[ "$resolved" != / && "$resolved" != "$HOME" ]] || fail unsafe_workspace 'historical workspace is too broad'
    printf '%s\n' "$resolved"
}

assert_architecture() {
    local architecture
    architecture="$(uname -m)"
    [[ "$architecture" == aarch64 || "$architecture" == arm64 ]] ||
        fail wrong_architecture "expected DGX Spark ARM64, found '$architecture'"
    printf 'architecture=%s\n' "$architecture"
}

find_disk_probe() {
    local probe="$1"
    while [[ ! -e "$probe" ]]; do
        local parent
        parent="$(dirname -- "$probe")"
        [[ "$parent" != "$probe" ]] || break
        probe="$parent"
    done
    printf '%s\n' "$probe"
}

check_disk_gib() {
    local workspace="$1"
    local minimum_gib="$2"
    require_uint min_free_disk_gib "$minimum_gib"
    local probe available_kib required_kib
    probe="$(find_disk_probe "$workspace")"
    available_kib="$(df -Pk -- "$probe" | awk 'NR == 2 {print $4}')"
    [[ "$available_kib" =~ ^[0-9]+$ ]] || fail disk_probe_failed 'could not read available disk space'
    required_kib=$((minimum_gib * 1024 * 1024))
    (( available_kib >= required_kib )) ||
        fail low_disk "need ${minimum_gib}GiB free; only $((available_kib / 1024 / 1024))GiB is available"
    printf 'disk_available_gib=%s\n' "$((available_kib / 1024 / 1024))"
}

available_memory_gib() {
    local available_kib
    available_kib="$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)"
    [[ "$available_kib" =~ ^[0-9]+$ ]] || fail memory_probe_failed 'could not read MemAvailable from /proc/meminfo'
    printf '%s\n' "$((available_kib / 1024 / 1024))"
}

report_available_disk_gib() {
    local workspace="$1"
    local probe available_kib
    probe="$(find_disk_probe "$workspace")"
    available_kib="$(df -Pk -- "$probe" | awk 'NR == 2 {print $4}')"
    [[ "$available_kib" =~ ^[0-9]+$ ]] || fail disk_probe_failed 'could not read available disk space'
    printf '%s\n' "$((available_kib / 1024 / 1024))"
}

summarize_training_progress() {
    local run_dir="$1"
    local metrics checkpoints names
    metrics="$run_dir/metrics.jsonl"
    checkpoints="$run_dir/checkpoints"
    if [[ -e "$metrics" || -L "$metrics" ]]; then
        assert_safe_regular_file "$metrics" "$run_dir" 'metrics stream'
        metrics="$DGX_SAFE_PATH"
        printf '%s\n' '--- latest metrics ---'
        python3 -I - "$metrics" <<'PY' || fail artifact_read_failed 'could not summarize metrics'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
latest = {}
count = 0
with path.open("r", encoding="utf-8") as handle:
    for raw in handle:
        line = raw.strip()
        if not line:
            continue
        record = json.loads(line)
        count += 1
        split = record.get("split")
        if split in ("train", "validation"):
            latest[split] = record
print(f"metrics_rows={count}")
for split in ("train", "validation"):
    prefix = f"latest_{split}"
    record = latest.get(split)
    if not isinstance(record, dict):
        print(f"{prefix}=absent")
        continue
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    loss = metrics.get("total_loss", metrics.get("loss"))
    print(f"{prefix}_step={record.get('step')}")
    print(f"{prefix}_loss={loss}")
    print(f"{prefix}_action_loss={metrics.get('action_loss')}")
    print(f"{prefix}_value_loss={metrics.get('value_loss')}")
    print(f"{prefix}_movement_exact_match={metrics.get('movement_exact_match')}")
    print(f"{prefix}_stage_index={metrics.get('stage_index')}")
PY
    else
        printf 'metrics=not-created\n'
    fi
    if [[ -e "$checkpoints" || -L "$checkpoints" ]]; then
        assert_safe_directory_path "$checkpoints" "$run_dir" 'checkpoint directory'
        checkpoints="$DGX_SAFE_PATH"
        printf '%s\n' '--- checkpoints ---'
        names="$(find "$checkpoints" -maxdepth 1 -type f -name 'step-*.pt' -printf '%f\n' | LC_ALL=C sort)"
        if [[ -z "$names" ]]; then
            printf 'checkpoints=none\n'
        else
            printf '%s\n' "$names"
        fi
    else
        printf 'checkpoints=not-created\n'
    fi
}

check_memory_gib() {
    local minimum_gib="$1"
    local container_limit_gib="${2:-0}"
    require_uint min_available_memory_gib "$minimum_gib"
    require_uint container_memory_gib "$container_limit_gib"
    local available_gib
    available_gib="$(available_memory_gib)"
    (( available_gib >= minimum_gib )) ||
        fail low_memory "need ${minimum_gib}GiB MemAvailable; only ${available_gib}GiB is available"
    if (( container_limit_gib > 0 )); then
        (( available_gib >= container_limit_gib + 4 )) ||
            fail low_memory "container limit ${container_limit_gib}GiB requires 4GiB headroom; only ${available_gib}GiB is available"
    fi
    printf 'memory_available_gib=%s\n' "$available_gib"
}

check_gpu_idle() {
    require_command nvidia-smi
    nvidia-smi

    local processes query_status
    set +e
    processes="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)"
    query_status=$?
    set -e
    (( query_status == 0 )) || fail gpu_process_probe_failed 'could not prove the GPU compute queue is idle'
    processes="$(printf '%s' "$processes" | sed '/^[[:space:]]*$/d')"
    [[ -z "$processes" ]] || fail gpu_busy "active GPU compute processes detected: $(printf '%s' "$processes" | tr '\n' ';')"
    printf 'gpu_compute_processes=none\n'
}

check_container_image() {
    local image="$1"
    require_image_reference "$image"
    require_command docker
    docker info >/dev/null 2>&1 || fail docker_unavailable 'Docker daemon is unavailable to this user'
    docker image inspect "$image" >/dev/null 2>&1 ||
        fail image_not_cached "image '$image' is not cached; automatic pulls are forbidden"
    DGX_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$image")" ||
        fail image_inspect_failed "could not resolve immutable ID for '$image'"
    require_image_id "$DGX_IMAGE_ID"
    printf 'image_id=%s\n' "$DGX_IMAGE_ID"
}

metadata_value() {
    local path="$1"
    local key="$2"
    [[ -f "$path" && ! -L "$path" ]] || fail invalid_run_metadata "metadata file '$path' is missing or unsafe"
    local -a matches=()
    mapfile -t matches < <(grep -E "^${key}=" "$path" || true)
    (( ${#matches[@]} == 1 )) || fail invalid_run_metadata "metadata key '$key' must occur exactly once"
    printf '%s\n' "${matches[0]#*=}"
}

require_file_contains() {
    local path="$1"
    local expected="$2"
    grep -F -- "$expected" "$path" >/dev/null ||
        fail runtime_boundary_mismatch "'$path' does not contain required boundary '$expected'"
}

assert_run_inactive() {
    local run_id="$1"
    local session="pseudo-brain-${run_id}"
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$session" 2>/dev/null; then
        fail run_active "tmux session '$session' is still active"
    fi
    local containers
    containers="$(docker ps -a --filter "name=^/pseudo-brain-${run_id}$" --format '{{.Names}} {{.Status}}')"
    [[ -z "$containers" ]] || fail run_active "container record for '$run_id' still exists: $containers"
}

assert_run_lock_available() {
    local run_dir="$1"
    local lock_path="$run_dir/.training.lock"
    [[ -f "$lock_path" && ! -L "$lock_path" ]] ||
        fail missing_run_lock 'run has no safe training lock and predates the resumable launch contract'
    (
        exec 9>> "$lock_path" || exit 75
        flock -n 9
    ) || fail run_lock_held 'the per-run training lock is held; refusing resume'
}

wait_for_detached_launch() {
    local session="$1"
    local ready_path="$2"
    local run_id="$3"
    local container_name="pseudo-brain-${run_id}"
    local attempt=0
    local containers
    while (( attempt < 50 )); do
        tmux has-session -t "$session" 2>/dev/null ||
            fail detached_launch_failed "tmux session '$session' exited before the training container became live"
        if [[ -e "$ready_path" || -L "$ready_path" ]]; then
            [[ -f "$ready_path" && ! -L "$ready_path" ]] ||
                fail invalid_launch_handshake "unsafe detached launch handshake '$ready_path'"
            grep -Fx 'lock_acquired=1' "$ready_path" >/dev/null ||
                fail invalid_launch_handshake 'detached launch handshake has invalid content'
            containers="$(docker ps --filter "name=^/${container_name}$" --format '{{.Names}}')" ||
                fail container_probe_failed 'could not verify the detached training container'
            if [[ "$containers" == "$container_name" ]]; then
                printf 'launch_handshake=%s\ncontainer=%s\n' "$ready_path" "$container_name"
                return 0
            fi
        fi
        sleep 0.1
        attempt=$((attempt + 1))
    done
    fail detached_launch_timeout "tmux session '$session' did not confirm lock acquisition"
}

assert_safe_directory_path() {
    local path="$1"
    local root="$2"
    local label="$3"
    local resolved_path resolved_root
    [[ -d "$root" && ! -L "$root" ]] || fail unsafe_artifact "$label root is missing or linked"
    [[ -d "$path" && ! -L "$path" ]] || fail unsafe_artifact "$label is missing or linked"
    resolved_root="$(realpath -e -- "$root")" || fail unsafe_artifact "could not resolve $label root"
    resolved_path="$(realpath -e -- "$path")" || fail unsafe_artifact "could not resolve $label"
    case "$resolved_path" in
        "$resolved_root"|"$resolved_root"/*) ;;
        *) fail artifact_escape "$label escaped its owned workspace root" ;;
    esac
    DGX_SAFE_PATH="$resolved_path"
}

assert_safe_regular_file() {
    local path="$1"
    local root="$2"
    local label="$3"
    [[ -f "$path" && ! -L "$path" ]] || fail unsafe_artifact "$label is missing or linked"
    assert_safe_directory_path "$(dirname -- "$path")" "$root" "$label parent"
    local resolved_parent="$DGX_SAFE_PATH"
    local resolved_path
    resolved_path="$(realpath -e -- "$path")" || fail unsafe_artifact "could not resolve $label"
    [[ "$(dirname -- "$resolved_path")" == "$resolved_parent" ]] ||
        fail artifact_escape "$label escaped its owned directory"
    DGX_SAFE_PATH="$resolved_path"
}

assert_account_owned_directory() {
    local path="$1"
    local root="$2"
    local label="$3"
    local expected_mode="${4:-}"
    local owner mode current_uid
    assert_safe_directory_path "$path" "$root" "$label"
    path="$DGX_SAFE_PATH"
    current_uid="$(id -u)" || fail identity_probe_failed 'could not read the account UID'
    owner="$(stat -c '%u' -- "$path")" || fail artifact_probe_failed "could not read $label ownership"
    mode="$(stat -c '%a' -- "$path")" || fail artifact_probe_failed "could not read $label mode"
    [[ "$owner" == "$current_uid" ]] || fail unsafe_artifact_owner "$label is not owned by the canonical account"
    if [[ -n "$expected_mode" ]]; then
        [[ "$mode" == "$expected_mode" ]] || fail unsafe_artifact_mode "$label must have mode $expected_mode"
    else
        (( (8#$mode & 8#022) == 0 )) || fail unsafe_artifact_mode "$label is group/other writable"
    fi
    DGX_SAFE_PATH="$path"
}

assert_account_owned_regular_file() {
    local path="$1"
    local root="$2"
    local label="$3"
    local owner mode links current_uid
    assert_safe_regular_file "$path" "$root" "$label"
    path="$DGX_SAFE_PATH"
    current_uid="$(id -u)" || fail identity_probe_failed 'could not read the account UID'
    read -r owner mode links < <(stat -c '%u %a %h' -- "$path") ||
        fail artifact_probe_failed "could not read $label ownership and mode"
    [[ "$owner" == "$current_uid" ]] || fail unsafe_artifact_owner "$label is not owned by the canonical account"
    (( (8#$mode & 8#022) == 0 )) || fail unsafe_artifact_mode "$label is group/other writable"
    [[ "$links" == 1 ]] || fail unsafe_artifact_link_count "$label must have exactly one hard link"
    DGX_SAFE_PATH="$path"
}

assert_account_owned_tree() {
    local path="$1"
    local root="$2"
    local label="$3"
    local current_uid unsafe_entry
    assert_safe_artifact_tree "$path" "$root" "$label"
    path="$DGX_SAFE_PATH"
    current_uid="$(id -u)" || fail identity_probe_failed 'could not read the account UID'
    unsafe_entry="$(find "$path" -mindepth 0 ! -user "$current_uid" -print -quit)" ||
        fail artifact_probe_failed "could not inspect $label ownership"
    [[ -z "$unsafe_entry" ]] || fail unsafe_artifact_owner "$label contains an entry owned by another account: $unsafe_entry"
    unsafe_entry="$(find "$path" -mindepth 0 -perm /022 -print -quit)" ||
        fail artifact_probe_failed "could not inspect $label modes"
    [[ -z "$unsafe_entry" ]] || fail unsafe_artifact_mode "$label contains a group/other-writable entry: $unsafe_entry"
    unsafe_entry="$(find "$path" -type f ! -links 1 -print -quit)" ||
        fail artifact_probe_failed "could not inspect $label link counts"
    [[ -z "$unsafe_entry" ]] || fail unsafe_artifact_link_count "$label contains a hardlinked file: $unsafe_entry"
    DGX_SAFE_PATH="$path"
}

resolve_canonical_rcq_v2_workspace() {
    local current_uid passwd_lines account_home resolved_home expected workspace marker marker_sha
    current_uid="$(id -u)" || fail identity_probe_failed 'could not read the account UID'
    passwd_lines="$(getent passwd "$current_uid")" ||
        fail account_home_probe_failed 'could not resolve the account home from the system account database'
    [[ -n "$passwd_lines" && "$passwd_lines" != *$'\n'* ]] ||
        fail account_home_probe_failed 'the account database did not return exactly one account'
    IFS=: read -r _ _ _ _ _ account_home _ <<< "$passwd_lines"
    [[ "$account_home" == /* && "$account_home" != *$'\n'* ]] ||
        fail account_home_probe_failed 'the system account home is not one absolute path'
    resolved_home="$(realpath -e -- "$account_home")" ||
        fail account_home_probe_failed 'the system account home cannot be resolved'
    [[ "$resolved_home" == "$account_home" ]] ||
        fail unsafe_workspace 'the system account home has a symlinked or non-canonical path'
    expected="$account_home/projects/pseudo-brain"
    [[ -d "$expected" && ! -L "$expected" ]] ||
        fail missing_workspace 'the canonical RCQ-v2 workspace does not exist'
    workspace="$(realpath -e -- "$expected")" || fail unsafe_workspace 'the canonical RCQ-v2 workspace cannot be resolved'
    [[ "$workspace" == "$expected" ]] ||
        fail unsafe_workspace 'the canonical RCQ-v2 workspace has a symlinked path or suffix'
    assert_account_owned_directory "$workspace" "$account_home/projects" 'canonical RCQ-v2 workspace'
    workspace="$DGX_SAFE_PATH"
    marker="$workspace/.pseudo-brain-workspace-v2"
    assert_account_owned_regular_file "$marker" "$workspace" 'canonical workspace marker'
    marker="$DGX_SAFE_PATH"
    marker_sha="$(sha256sum "$marker" | awk '{print $1}')" ||
        fail marker_probe_failed 'could not hash the canonical workspace marker'
    [[ "$marker_sha" == "$RCQ_WORKSPACE_MARKER_SHA256" ]] ||
        fail unowned_workspace 'the canonical workspace marker bytes are not exact'
    DGX_SAFE_PATH="$workspace"
    printf '%s\n' "$workspace"
}

ensure_private_directory() {
    local path="$1"
    local root="$2"
    local label="$3"
    if [[ ! -e "$path" && ! -L "$path" ]]; then
        (umask 077; mkdir -- "$path") || fail authority_directory_create_failed "could not create $label"
    fi
    assert_account_owned_directory "$path" "$root" "$label" 700
}

open_private_lock() {
    local path="$1"
    local root="$2"
    local label="$3"
    local descriptor="${4:-9}"
    local path_identity descriptor_identity
    [[ "$descriptor" == 8 || "$descriptor" == 9 ]] || fail invalid_lock_descriptor 'authority lock descriptor must be 8 or 9'
    if [[ ! -e "$path" && ! -L "$path" ]]; then
        (umask 077; set -o noclobber; : > "$path") ||
            fail authority_lock_create_failed "could not create $label"
    fi
    assert_account_owned_regular_file "$path" "$root" "$label"
    path="$DGX_SAFE_PATH"
    [[ "$(stat -c '%a' -- "$path")" == 600 ]] || fail unsafe_artifact_mode "$label must have mode 600"
    path_identity="$(stat -Lc '%d:%i' -- "$path")" || fail artifact_probe_failed "could not identify $label"
    if [[ "$descriptor" == 8 ]]; then
        exec 8<> "$path" || fail authority_lock_open_failed "could not open $label"
    else
        exec 9<> "$path" || fail authority_lock_open_failed "could not open $label"
    fi
    descriptor_identity="$(stat -Lc '%d:%i' -- "/proc/$$/fd/$descriptor")" ||
        fail authority_lock_open_failed "could not identify the opened $label"
    [[ "$descriptor_identity" == "$path_identity" ]] || fail authority_lock_race "$label changed while it was opened"
    flock -n "$descriptor" || fail authority_busy "$label is already held by another qualification action"
    DGX_SAFE_PATH="$path"
}

assert_safe_artifact_tree() {
    local path="$1"
    local root="$2"
    local label="$3"
    local unsafe_entry
    assert_safe_directory_path "$path" "$root" "$label"
    unsafe_entry="$(find "$DGX_SAFE_PATH" -mindepth 1 \! -type d \! -type f -print -quit)" ||
        fail artifact_probe_failed "could not inspect $label"
    [[ -z "$unsafe_entry" ]] ||
        fail unsafe_artifact_tree "$label contains a link or special file: $unsafe_entry"
}

assert_clean_release_python_source() {
    local release="$1"
    local source_root package_root root_entries unsafe_entry
    source_root="$release/brain/src"
    assert_safe_directory_path "$source_root" "$release" 'release Python source root'
    source_root="$DGX_SAFE_PATH"
    root_entries="$(find "$source_root" -mindepth 1 -maxdepth 1 -printf '%f\n')" ||
        fail source_tree_probe_failed 'could not inspect release Python source root'
    [[ "$root_entries" == irene_brain ]] ||
        fail unsafe_source_tree 'brain/src must contain exactly the irene_brain package directory'

    package_root="$source_root/irene_brain"
    [[ -d "$package_root" && ! -L "$package_root" ]] ||
        fail unsafe_source_tree 'irene_brain package root is missing or linked'
    unsafe_entry="$(find "$package_root" -mindepth 1 \
        \( -name __pycache__ -o ! \( -type d -o \( -type f -name '*.py' -links 1 \) \) \) \
        -print -quit 2>/dev/null)" ||
        fail source_tree_probe_failed 'could not inspect release Python package entries'
    [[ -z "$unsafe_entry" ]] ||
        fail unsafe_source_tree "release Python package contains a non-.py, linked, special, hardlinked, or cache entry: $unsafe_entry"
    [[ -f "$package_root/__init__.py" && ! -L "$package_root/__init__.py" ]] ||
        fail unsafe_source_tree 'irene_brain package has no safe __init__.py'
    DGX_SAFE_PATH="$source_root"
}

assert_canonical_rcq_v2_registration_file() {
    local path="$1"
    local root="$2"
    assert_safe_regular_file "$path" "$root" 'fixed RCQ-v2 registration'
    path="$DGX_SAFE_PATH"
    local size
    size="$(wc -c < "$path")" || fail registration_probe_failed 'could not size the fixed RCQ-v2 registration'
    require_uint registration_size_bytes "$size"
    (( size > 0 && size <= 1048576 )) ||
        fail invalid_registration 'fixed RCQ-v2 registration size is outside the safe bound'
    python3 -I - "$path" <<'PY' || fail invalid_registration 'fixed RCQ-v2 registration is not one canonical target-blind object'
import json
from pathlib import Path
import sys


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


path = Path(sys.argv[1])
encoded = path.read_bytes()
if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
    raise ValueError("registration must contain one newline-terminated JSON object")
value = json.loads(
    encoded[:-1].decode("utf-8", errors="strict"),
    object_pairs_hook=strict_object,
    parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {token}")
    ),
)
canonical = (
    json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
).encode("utf-8")
if encoded != canonical:
    raise ValueError("registration bytes are not canonical")
if not isinstance(value, dict):
    raise ValueError("registration root is not an object")
if value.get("schema_version") != 2:
    raise ValueError("registration schema is not RCQ-v2 schema 2")
if value.get("qualification_id") != "rcq_v2_reference_v2":
    raise ValueError("registration qualification is not fixed")
protocol = value.get("workspace_protocol")
if not isinstance(protocol, dict) or protocol.get("contract") != "pseudo-brain-workspace-v2":
    raise ValueError("registration workspace protocol is not fixed")
if protocol.get("registration_release_relative_path") != "registrations/rcq-v2-reference-v2.json":
    raise ValueError("registration path contract is not fixed")
PY
    DGX_SAFE_PATH="$path"
}

load_rcq_v2_pretraining_pin_summary() {
    local workspace="$1"
    local pin_parent pin_dir pin_path
    pin_parent="$workspace/qualification-pins"
    assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
    pin_parent="$DGX_SAFE_PATH"
    pin_dir="$pin_parent/rcq-v2-reference-v2"
    assert_account_owned_directory "$pin_dir" "$pin_parent" 'RCQ-v2 qualification pin directory' 700
    pin_dir="$DGX_SAFE_PATH"
    pin_path="$pin_dir/$RCQ_PRETRAINING_PIN_FILENAME"
    assert_account_owned_regular_file "$pin_path" "$pin_dir" 'RCQ-v2 pretraining pin'
    pin_path="$DGX_SAFE_PATH"
    [[ "$(stat -c '%a' -- "$pin_path")" == 400 ]] ||
        fail unsafe_artifact_mode 'RCQ-v2 pretraining pin must have exact mode 0400'
    python3 -I - "$pin_path" <<'PY' || fail invalid_pretraining_pin 'RCQ-v2 pretraining pin failed strict validation'
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re
import sys


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} fields differ")


def hash_string(value, name):
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"{name} is not one lowercase SHA-256")
    return value


path = sys.argv[1]
encoded = open(path, "rb").read()
if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
    raise ValueError("pin is not one newline-terminated object")
value = json.loads(
    encoded[:-1].decode("utf-8", errors="strict"),
    object_pairs_hook=strict_object,
    parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
)
if encoded != (canonical(value) + "\n").encode("utf-8"):
    raise ValueError("pin bytes are not canonical")
exact_keys(
    value,
    {
        "schema_version", "action", "qualification_id", "workspace", "release",
        "registration", "config", "source_tree_sha256", "evaluator_bundle_sha256",
        "batch_source_manifest_sha256", "runtime", "run", "range_claim_id",
        "created_utc", "pin_sha256",
    },
    "pretraining pin",
)
if value["schema_version"] != 1 or value["action"] != "rcq_v2_pin_pretraining_v1":
    raise ValueError("pretraining pin identity differs")
if value["qualification_id"] != "rcq_v2_reference_v2":
    raise ValueError("pretraining qualification differs")
workspace = value["workspace"]
exact_keys(
    workspace,
    {
        "contract", "host_account_home_relative_path", "marker_relative_path",
        "marker_file_sha256", "claim_registry_relative_path",
        "pin_directory_relative_path",
    },
    "workspace",
)
if workspace != {
    "contract": "pseudo-brain-workspace-v2",
    "host_account_home_relative_path": "projects/pseudo-brain",
    "marker_relative_path": ".pseudo-brain-workspace-v2",
    "marker_file_sha256": "5657bd63ee6ff61d1d90ea63b4cb02f9dd7d5dc3df223c9cbb028f9e2a096053",
    "claim_registry_relative_path": "final-claims",
    "pin_directory_relative_path": "qualification-pins/rcq-v2-reference-v2",
}:
    raise ValueError("workspace binding differs")
release = value["release"]
exact_keys(release, {"id", "relative_path", "archive_sha256"}, "release")
release_id = release["id"]
if type(release_id) is not str or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,62}", release_id) is None:
    raise ValueError("release id is unsafe")
if release["relative_path"] != f"releases/{release_id}":
    raise ValueError("release relative path differs")
archive_sha = hash_string(release["archive_sha256"], "archive_sha256")
registration = value["registration"]
exact_keys(registration, {"release_relative_path", "sha256"}, "registration")
if registration["release_relative_path"] != "registrations/rcq-v2-reference-v2.json":
    raise ValueError("registration path differs")
registration_sha = hash_string(registration["sha256"], "registration_sha256")
config = value["config"]
exact_keys(config, {"release_relative_path", "raw_sha256", "canonical_sha256"}, "config")
if config["release_relative_path"] != "brain/configs/training/dgx-rcq-v2-reference.toml":
    raise ValueError("config path differs")
config_raw_sha = hash_string(config["raw_sha256"], "config_raw_sha256")
hash_string(config["canonical_sha256"], "config_canonical_sha256")
hash_string(value["source_tree_sha256"], "source_tree_sha256")
hash_string(value["evaluator_bundle_sha256"], "evaluator_bundle_sha256")
hash_string(value["batch_source_manifest_sha256"], "batch_source_manifest_sha256")
runtime = value["runtime"]
exact_keys(
    runtime,
    {
        "container_image_reference", "container_image_id", "container_release_root",
        "container_run_root", "container_claim_registry_root", "container_pin_root",
        "network",
    },
    "runtime",
)
image_reference = runtime["container_image_reference"]
image_id = runtime["container_image_id"]
if type(image_reference) is not str or not image_reference or any(c.isspace() for c in image_reference):
    raise ValueError("container image reference is unsafe")
if type(image_id) is not str or re.fullmatch(r"sha256:[a-f0-9]{64}", image_id) is None:
    raise ValueError("container image id is not immutable")
if runtime | {"container_image_reference": image_reference, "container_image_id": image_id} != {
    "container_image_reference": image_reference,
    "container_image_id": image_id,
    "container_release_root": "/workspace/repo",
    "container_run_root": "/workspace/run",
    "container_claim_registry_root": "/workspace/final-claims",
    "container_pin_root": "/workspace/pins",
    "network": "none",
}:
    raise ValueError("container runtime binding differs")
run = value["run"]
if run != {
    "id": "dgx-rcq-v2-reference-seed-1702",
    "relative_path": "runs/dgx-rcq-v2-reference-seed-1702",
    "seed": 1702,
    "final_step": 2048,
}:
    raise ValueError("fixed run binding differs")
range_claim_id = hash_string(value["range_claim_id"], "range_claim_id")
if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value["created_utc"]) is None:
    raise ValueError("created_utc is not strict RFC3339 UTC")
pin_sha = hash_string(value["pin_sha256"], "pin_sha256")
semantic = dict(value)
del semantic["pin_sha256"]
expected_pin_sha = sha256(
    b"PSEUDOBRAINRCQPRETRAINPIN\x01" + canonical(semantic).encode("utf-8")
).hexdigest()
if pin_sha != expected_pin_sha:
    raise ValueError("pretraining semantic digest differs")
for relative in (
    release["relative_path"], registration["release_relative_path"],
    config["release_relative_path"], run["relative_path"],
):
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError("pin contains an unsafe relative path")
print("\t".join((
    release_id, archive_sha, registration_sha, config_raw_sha,
    image_reference, image_id, range_claim_id, pin_sha,
)))
PY
}

verify_rcq_v2_pretraining_host_bindings() {
    local workspace="$1"
    local release_id="$2"
    local expected_archive_sha="$3"
    local expected_registration_sha="$4"
    local expected_config_sha="$5"
    local image_reference="$6"
    local expected_image_id="$7"
    local range_claim_id="$8"
    local release_root release archive_receipt registration config claim_root claim_leaf actual
    require_slug release_id "$release_id"
    require_sha256 release_archive_sha256 "$expected_archive_sha"
    require_sha256 registration_sha256 "$expected_registration_sha"
    require_sha256 config_raw_sha256 "$expected_config_sha"
    require_image_reference "$image_reference"
    require_image_id "$expected_image_id"
    require_sha256 range_claim_id "$range_claim_id"

    release_root="$workspace/releases"
    assert_account_owned_directory "$release_root" "$workspace" 'release root'
    release_root="$DGX_SAFE_PATH"
    release="$release_root/$release_id"
    assert_account_owned_tree "$release" "$release_root" 'pinned release tree'
    release="$DGX_SAFE_PATH"
    assert_clean_release_python_source "$release"
    archive_receipt="$release/.source-archive.sha256"
    assert_account_owned_regular_file "$archive_receipt" "$release" 'release archive receipt'
    archive_receipt="$DGX_SAFE_PATH"
    [[ "$(wc -c < "$archive_receipt")" == 65 && "$(wc -l < "$archive_receipt")" == 1 ]] ||
        fail invalid_release_receipt 'release archive receipt is not one lowercase SHA-256 line'
    actual="$(sed -n '1p' "$archive_receipt")"
    [[ "$actual" == "$expected_archive_sha" ]] || fail release_archive_hash_mismatch 'pinned release archive changed'
    registration="$release/$RCQ_REGISTRATION_RELATIVE_PATH"
    assert_canonical_rcq_v2_registration_file "$registration" "$release"
    registration="$DGX_SAFE_PATH"
    actual="$(sha256sum "$registration" | awk '{print $1}')"
    [[ "$actual" == "$expected_registration_sha" ]] || fail registration_hash_mismatch 'pinned registration changed'
    config="$release/$RCQ_REFERENCE_CONFIG"
    assert_account_owned_regular_file "$config" "$release" 'fixed RCQ-v2 training configuration'
    config="$DGX_SAFE_PATH"
    actual="$(sha256sum "$config" | awk '{print $1}')"
    [[ "$actual" == "$expected_config_sha" ]] || fail config_hash_mismatch 'pinned training configuration changed'
    check_container_image "$image_reference"
    [[ "$DGX_IMAGE_ID" == "$expected_image_id" ]] || fail image_id_mismatch 'pinned immutable container image is unavailable or changed'

    claim_root="$workspace/final-claims"
    assert_account_owned_directory "$claim_root" "$workspace" 'persistent final-claim registry' 700
    claim_root="$DGX_SAFE_PATH"
    assert_account_owned_tree "$claim_root" "$workspace" 'persistent final-claim registry tree'
    claim_root="$DGX_SAFE_PATH"
    claim_leaf="$claim_root/$range_claim_id"
    [[ ! -e "$claim_leaf" && ! -L "$claim_leaf" ]] ||
        fail range_already_claimed 'the registered final TEST range is already retired in this workspace'
    DGX_SAFE_PATH="$release"
}

bind_rcq_v2_pretraining_authority() {
    local workspace="$1"
    local summary pin_parent pin_dir pin_path
    summary="$(load_rcq_v2_pretraining_pin_summary "$workspace")" ||
        fail invalid_pretraining_pin 'could not load the RCQ-v2 pretraining trust root'
    IFS=$'\t' read -r \
        RCQ_PIN_RELEASE_ID RCQ_PIN_ARCHIVE_SHA256 RCQ_PIN_REGISTRATION_SHA256 \
        RCQ_PIN_CONFIG_SHA256 RCQ_PIN_IMAGE_REFERENCE RCQ_PIN_IMAGE_ID \
        RCQ_PIN_RANGE_CLAIM_ID RCQ_PIN_SEMANTIC_SHA256 <<< "$summary"
    for value in \
        "$RCQ_PIN_RELEASE_ID" "$RCQ_PIN_ARCHIVE_SHA256" "$RCQ_PIN_REGISTRATION_SHA256" \
        "$RCQ_PIN_CONFIG_SHA256" "$RCQ_PIN_IMAGE_REFERENCE" "$RCQ_PIN_IMAGE_ID" \
        "$RCQ_PIN_RANGE_CLAIM_ID" "$RCQ_PIN_SEMANTIC_SHA256"; do
        [[ -n "$value" ]] || fail invalid_pretraining_pin 'pretraining pin summary is incomplete'
    done
    pin_parent="$workspace/qualification-pins"
    assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
    pin_parent="$DGX_SAFE_PATH"
    pin_dir="$pin_parent/rcq-v2-reference-v2"
    assert_account_owned_directory "$pin_dir" "$pin_parent" 'RCQ-v2 qualification pin directory' 700
    RCQ_PIN_DIRECTORY="$DGX_SAFE_PATH"
    pin_path="$RCQ_PIN_DIRECTORY/$RCQ_PRETRAINING_PIN_FILENAME"
    assert_account_owned_regular_file "$pin_path" "$RCQ_PIN_DIRECTORY" 'RCQ-v2 pretraining pin'
    RCQ_PIN_PRETRAINING_PATH="$DGX_SAFE_PATH"
    [[ "$(stat -c '%a' -- "$RCQ_PIN_PRETRAINING_PATH")" == 400 ]] ||
        fail unsafe_artifact_mode 'RCQ-v2 pretraining pin must have exact mode 0400'
    RCQ_PIN_FILE_SHA256="$(sha256sum "$RCQ_PIN_PRETRAINING_PATH" | awk '{print $1}')" ||
        fail pretraining_pin_probe_failed 'could not hash the RCQ-v2 pretraining pin'
    require_sha256 pretraining_pin_file_sha256 "$RCQ_PIN_FILE_SHA256"
    verify_rcq_v2_pretraining_host_bindings \
        "$workspace" "$RCQ_PIN_RELEASE_ID" "$RCQ_PIN_ARCHIVE_SHA256" \
        "$RCQ_PIN_REGISTRATION_SHA256" "$RCQ_PIN_CONFIG_SHA256" \
        "$RCQ_PIN_IMAGE_REFERENCE" "$RCQ_PIN_IMAGE_ID" "$RCQ_PIN_RANGE_CLAIM_ID"
    RCQ_PIN_RELEASE_PATH="$DGX_SAFE_PATH"
}

refuse_terminal_failed_stage_gate() {
    local run_dir="$1"
    local checkpoint_step="$2"
    local release="$3"
    local semantic_root kind path entry_path gate_status parse_status
    require_uint checkpoint_step "$checkpoint_step"

    assert_clean_release_python_source "$release"
    semantic_root="$DGX_SAFE_PATH"

    for kind in development-gate final-development; do
        path="$run_dir/${kind}-step-$(printf '%08d' "$checkpoint_step").json"
        if [[ ! -e "$path" && ! -L "$path" ]]; then
            continue
        fi
        assert_safe_regular_file "$path" "$run_dir" "terminal $kind artifact"
        path="$DGX_SAFE_PATH"
        entry_path='none'
        if [[ "$kind" == development-gate ]]; then
            (( checkpoint_step == 1536 )) ||
                fail invalid_terminal_gate_artifact 'registered transition gate is not at optimizer step 1536'
        else
            (( checkpoint_step == 2048 )) ||
                fail invalid_terminal_gate_artifact 'registered completion gate is not at optimizer step 2048'
            entry_path="$run_dir/development-gate-step-00001536.json"
            assert_safe_regular_file "$entry_path" "$run_dir" \
                'completion entry development-gate artifact'
            entry_path="$DGX_SAFE_PATH"
        fi
        set +e
        gate_status="$(python3 -I - "$semantic_root" "$path" "$kind" "$entry_path" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_canonical(path, *, label):
    with open(path, "rb") as stream:
        encoded = stream.read()
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ValueError(f"{label} is not one newline-terminated object")
    value = json.loads(
        encoded[:-1].decode("utf-8", errors="strict"),
        object_pairs_hook=strict_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {token}")
        ),
    )
    canonical = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if encoded != canonical:
        raise ValueError(f"{label} is not byte-canonical")
    return value, encoded


source_root = Path(sys.argv[1]).resolve(strict=True)
artifact_path = Path(sys.argv[2])
kind = sys.argv[3]
entry_path = sys.argv[4]
sys.dont_write_bytecode = True
sys.path.insert(0, str(source_root))

from irene_brain.evaluation.rcq_v2 import (  # noqa: E402
    DEVELOPMENT_SAMPLES,
    evaluate_rcq_v2_development,
)
from irene_brain.training.protocol import TrainingStepResult  # noqa: E402


value, encoded = load_canonical(artifact_path, label=f"terminal {kind} artifact")
if not isinstance(value, dict) or not isinstance(value.get("metrics"), dict):
    raise ValueError(f"terminal {kind} artifact has invalid metrics")
schemas = {
    "development-gate": (
        {"schema_version", "gate", "scope", "passed", "checks", "metrics"},
        "rcq_v2_development_v1",
    ),
    "final-development": (
        {
            "schema_version",
            "gate",
            "scope",
            "passed",
            "entry_gate",
            "registered_value_baselines",
            "checks",
            "metrics",
        },
        "rcq_v2_value_development_v1",
    ),
}
try:
    required_fields, expected_gate = schemas[kind]
except KeyError as error:
    raise ValueError("terminal gate kind is unsupported") from error
if set(value) != required_fields or value.get("gate") != expected_gate:
    raise ValueError(f"terminal {kind} artifact has an incompatible gate schema")
result = TrainingStepResult(
    loss=0.0,
    metrics=value["metrics"],
    samples=DEVELOPMENT_SAMPLES,
)

if kind == "development-gate":
    if entry_path != "none":
        raise ValueError("transition validation received an entry artifact")
    report = evaluate_rcq_v2_development(result)
elif kind == "final-development":
    from irene_brain.evaluation.rcq_v2 import (  # noqa: E402
        evaluate_rcq_v2_value_development,
    )

    if entry_path == "none":
        raise ValueError("completion validation requires its entry artifact")
    entry_value, entry_encoded = load_canonical(
        Path(entry_path),
        label="completion entry development-gate artifact",
    )
    if not isinstance(entry_value, dict) or not isinstance(
        entry_value.get("metrics"), dict
    ):
        raise ValueError("completion entry development-gate metrics are invalid")
    entry_report = evaluate_rcq_v2_development(
        TrainingStepResult(
            loss=0.0,
            metrics=entry_value["metrics"],
            samples=DEVELOPMENT_SAMPLES,
        )
    )
    expected_entry = (entry_report.canonical_json + "\n").encode("utf-8")
    if entry_encoded != expected_entry:
        raise ValueError("completion entry development-gate artifact is inconsistent")
    report = evaluate_rcq_v2_value_development(
        result,
        entry_report=entry_report,
        entry_report_sha256=sha256(entry_encoded).hexdigest(),
    )
else:
    raise AssertionError("unreachable terminal gate kind")

expected = (report.canonical_json + "\n").encode("utf-8")
if encoded != expected:
    raise ValueError(f"terminal {kind} artifact failed semantic reconstruction")
print("passed" if report.passed else "failed")
PY
)"
        parse_status=$?
        set -e
        (( parse_status == 0 )) ||
            fail invalid_terminal_gate_artifact "terminal $kind artifact failed strict validation"
        if [[ "$gate_status" == failed ]]; then
            if [[ "$kind" == development-gate ]]; then
                fail terminal_transition_gate_failed 'selected checkpoint is terminal because its registered development transition gate failed'
            fi
            fail terminal_completion_gate_failed 'selected checkpoint is terminal because its registered final development gate failed'
        fi
        [[ "$gate_status" == passed ]] ||
            fail invalid_terminal_gate_artifact "terminal $kind artifact returned an invalid status"
    done
}

require_rcq_staging_canary_receipt() {
    local workspace="$1"
    local release="$2"
    local release_id="$3"
    local image="$4"
    local image_id="$5"
    local expected_pretraining_file_sha="$6"
    local expected_pretraining_semantic_sha="$7"
    local receipt_root receipt receipt_lines config_sha run_id run_root run_dir artifact
    local final_checkpoint final_checkpoint_sha metrics_sha transition_sha invariance_sha

    receipt_root="$workspace/rcq-staging-canary-receipts"
    assert_safe_directory_path "$receipt_root" "$workspace" 'staging-canary receipt root'
    receipt_root="$DGX_SAFE_PATH"
    receipt="$receipt_root/${release_id}.env"
    assert_safe_regular_file "$receipt" "$receipt_root" 'staging-canary receipt'
    receipt="$DGX_SAFE_PATH"
    receipt_lines="$(awk 'END {print NR}' "$receipt")" ||
        fail invalid_staging_canary_receipt 'could not count staging-canary receipt fields'
    [[ "$receipt_lines" == 16 ]] ||
        fail invalid_staging_canary_receipt 'staging-canary receipt has an unexpected field count'
    [[ "$(metadata_value "$receipt" workspace_contract)" == pseudo-brain-workspace-v2 ]] ||
        fail invalid_staging_canary_receipt 'staging canary used a different workspace contract'
    [[ "$(metadata_value "$receipt" canary_contract)" == rcq-schema3-two-phase-v1 ]] ||
        fail invalid_staging_canary_receipt 'staging canary used a different execution contract'
    [[ "$(metadata_value "$receipt" release_id)" == "$release_id" ]] ||
        fail invalid_staging_canary_receipt 'staging canary used a different release'
    [[ "$(metadata_value "$receipt" container_image)" == "$image" ]] ||
        fail staging_canary_image_mismatch 'staging canary used a different container image'
    [[ "$(metadata_value "$receipt" container_image_id)" == "$image_id" ]] ||
        fail staging_canary_image_mismatch 'cached image content changed since the staging canary'
    require_sha256 pretraining_pin_file_sha256 "$expected_pretraining_file_sha"
    require_sha256 pretraining_pin_sha256 "$expected_pretraining_semantic_sha"
    [[ "$(metadata_value "$receipt" pretraining_pin_file_sha256)" == "$expected_pretraining_file_sha" ]] ||
        fail invalid_staging_canary_receipt 'staging canary used a different pretraining pin file'
    [[ "$(metadata_value "$receipt" pretraining_pin_sha256)" == "$expected_pretraining_semantic_sha" ]] ||
        fail invalid_staging_canary_receipt 'staging canary used a different pretraining semantic pin'
    [[ "$(metadata_value "$receipt" config_path)" == "$RCQ_STAGING_CANARY_CONFIG" ]] ||
        fail invalid_staging_canary_receipt 'staging canary used an unexpected configuration path'
    config_sha="$(sha256sum "$release/$RCQ_STAGING_CANARY_CONFIG" | awk '{print $1}')" ||
        fail config_hash_failed 'could not hash the staging-canary configuration'
    [[ "$(metadata_value "$receipt" config_sha256)" == "$config_sha" ]] ||
        fail invalid_staging_canary_receipt 'staging-canary configuration bytes changed'
    [[ "$(metadata_value "$receipt" final_checkpoint)" == step-00000002.pt ]] ||
        fail invalid_staging_canary_receipt 'staging canary did not finish its exact two-update budget'

    run_id="$(metadata_value "$receipt" run_id)"
    require_slug canary_run_id "$run_id"
    run_root="$workspace/runs"
    assert_safe_directory_path "$run_root" "$workspace" 'run root'
    run_root="$DGX_SAFE_PATH"
    run_dir="$run_root/$run_id"
    assert_safe_directory_path "$run_dir" "$run_root" 'staging-canary run'
    run_dir="$DGX_SAFE_PATH"
    final_checkpoint="$run_dir/checkpoints/step-00000002.pt"
    for artifact in \
        "$final_checkpoint" \
        "$run_dir/metrics.jsonl" \
        "$run_dir/stage-transition-01.json" \
        "$run_dir/invariance-step-00000002.json"; do
        assert_safe_regular_file "$artifact" "$run_dir" 'staging-canary evidence artifact'
    done
    final_checkpoint_sha="$(metadata_value "$receipt" final_checkpoint_sha256)"
    metrics_sha="$(metadata_value "$receipt" metrics_sha256)"
    transition_sha="$(metadata_value "$receipt" stage_transition_sha256)"
    invariance_sha="$(metadata_value "$receipt" final_invariance_sha256)"
    require_sha256 final_checkpoint_sha256 "$final_checkpoint_sha"
    require_sha256 metrics_sha256 "$metrics_sha"
    require_sha256 stage_transition_sha256 "$transition_sha"
    require_sha256 final_invariance_sha256 "$invariance_sha"
    [[ "$(sha256sum "$final_checkpoint" | awk '{print $1}')" == "$final_checkpoint_sha" ]] ||
        fail invalid_staging_canary_receipt 'staging-canary final checkpoint changed after qualification'
    [[ "$(sha256sum "$run_dir/metrics.jsonl" | awk '{print $1}')" == "$metrics_sha" ]] ||
        fail invalid_staging_canary_receipt 'staging-canary metrics changed after qualification'
    [[ "$(sha256sum "$run_dir/stage-transition-01.json" | awk '{print $1}')" == "$transition_sha" ]] ||
        fail invalid_staging_canary_receipt 'staging-canary transition evidence changed after qualification'
    [[ "$(sha256sum "$run_dir/invariance-step-00000002.json" | awk '{print $1}')" == "$invariance_sha" ]] ||
        fail invalid_staging_canary_receipt 'staging-canary invariance evidence changed after qualification'
    metadata_value "$receipt" completed_utc >/dev/null
}

verify_original_run_boundaries() {
    local workspace="$1"
    local release_id="$2"
    local image="$3"
    local config_path="$4"
    local run_id="$5"
    local cpu_limit="$6"
    local memory_limit_gib="$7"
    local current_image_id="$8"
    local min_disk_gib="$9"
    local min_memory_gib="${10}"
    local container_uid="${11}"
    local container_gid="${12}"
    local release_root release run_root run_dir dataset_root metadata launch expected_config_sha recorded_launch_sha

    release_root="$workspace/releases"
    assert_safe_directory_path "$release_root" "$workspace" 'release root'
    release_root="$DGX_SAFE_PATH"
    release="$release_root/$release_id"
    assert_safe_directory_path "$release" "$release_root" 'source release'
    release="$DGX_SAFE_PATH"
    assert_clean_release_python_source "$release"
    run_root="$workspace/runs"
    assert_safe_directory_path "$run_root" "$workspace" 'run root'
    run_root="$DGX_SAFE_PATH"
    run_dir="$run_root/$run_id"
    assert_safe_directory_path "$run_dir" "$run_root" 'owned run'
    run_dir="$DGX_SAFE_PATH"
    dataset_root="$workspace/datasets"
    assert_safe_directory_path "$dataset_root" "$workspace" 'dataset root'
    dataset_root="$DGX_SAFE_PATH"
    metadata="$run_dir/run.env"
    launch="$run_dir/launch.sh"
    [[ -d "$run_dir" && ! -L "$run_dir" ]] || fail missing_run "owned run '$run_id' does not exist"
    [[ -f "$launch" && ! -L "$launch" ]] || fail invalid_run_metadata 'original launch script is missing or unsafe'
    [[ -f "$run_dir/resolved-config.json" && ! -L "$run_dir/resolved-config.json" ]] ||
        fail invalid_run_metadata 'resolved training configuration is missing or unsafe'

    [[ "$(metadata_value "$metadata" release_id)" == "$release_id" ]] ||
        fail run_identity_mismatch 'release ID differs from the original run'
    [[ "$(metadata_value "$metadata" workspace_contract)" == pseudo-brain-workspace-v2 ]] ||
        fail run_identity_mismatch 'workspace contract differs from Pseudo-Brain v2'
    [[ "$(metadata_value "$metadata" container_image)" == "$image" ]] ||
        fail run_identity_mismatch 'container image differs from the original run'
    [[ "$(metadata_value "$metadata" container_image_id)" == "$current_image_id" ]] ||
        fail run_identity_mismatch 'container image content differs from the original run'
    [[ "$(metadata_value "$metadata" config_path)" == "$config_path" ]] ||
        fail run_identity_mismatch 'configuration path differs from the original run'
    [[ "$(metadata_value "$metadata" container_cpu_count)" == "$cpu_limit" ]] ||
        fail run_identity_mismatch 'container CPU limit differs from the original run'
    [[ "$(metadata_value "$metadata" container_memory_gib)" == "$memory_limit_gib" ]] ||
        fail run_identity_mismatch 'container memory limit differs from the original run'
    [[ "$(metadata_value "$metadata" min_free_disk_gib)" == "$min_disk_gib" ]] ||
        fail run_identity_mismatch 'minimum free-disk gate differs from the original run'
    [[ "$(metadata_value "$metadata" min_available_memory_gib)" == "$min_memory_gib" ]] ||
        fail run_identity_mismatch 'minimum available-memory gate differs from the original run'
    [[ "$(metadata_value "$metadata" container_uid)" == "$container_uid" ]] ||
        fail run_identity_mismatch 'container UID differs from the original run'
    [[ "$(metadata_value "$metadata" container_gid)" == "$container_gid" ]] ||
        fail run_identity_mismatch 'container GID differs from the original run'
    [[ "$(metadata_value "$metadata" network)" == none ]] ||
        fail run_identity_mismatch 'container network boundary differs from the original run'
    [[ "$(metadata_value "$metadata" ipc)" == host ]] ||
        fail run_identity_mismatch 'container IPC boundary differs from the original run'
    [[ "$(metadata_value "$metadata" pids_limit)" == 2048 ]] ||
        fail run_identity_mismatch 'container PID limit differs from the original run'
    [[ "$(metadata_value "$metadata" cublas_workspace_config)" == :4096:8 ]] ||
        fail run_identity_mismatch 'deterministic CUDA boundary differs from the original run'
    expected_config_sha="$(sha256sum "$release/$config_path" | awk '{print $1}')"
    [[ "$(metadata_value "$metadata" config_sha256)" == "$expected_config_sha" ]] ||
        fail run_identity_mismatch 'configuration bytes differ from the original run'

    require_file_contains "$launch" '--gpus all'
    require_file_contains "$launch" '--network none'
    require_file_contains "$launch" '--ipc host'
    require_file_contains "$launch" "--cpus \"$cpu_limit\""
    require_file_contains "$launch" "--memory \"${memory_limit_gib}g\""
    require_file_contains "$launch" '--pids-limit 2048'
    require_file_contains "$launch" '--stop-timeout 30'
    require_file_contains "$launch" "--user \"$container_uid:$container_gid\""
    require_file_contains "$launch" '#!/bin/bash'
    require_file_contains "$launch" "PATH='/usr/sbin:/usr/bin'"
    require_file_contains "$launch" 'unset BASH_ENV ENV PYTHONHOME PYTHONPATH CDPATH'
    grep -F -- '-e PYTHONPATH=' "$launch" >/dev/null &&
        fail runtime_boundary_mismatch 'original launch used a non-isolated PYTHONPATH bootstrap'
    require_file_contains "$launch" '-e HOME=/workspace/run'
    require_file_contains "$launch" '-e PATH=/usr/sbin:/usr/bin'
    require_file_contains "$launch" '-e PSEUDO_BRAIN_RUN_DIR=/workspace/run'
    grep -F -- 'IRENE_BRAIN_' "$launch" >/dev/null &&
        fail runtime_boundary_mismatch 'original authoritative launch used a legacy Irene environment name'
    require_file_contains "$launch" '-e CUBLAS_WORKSPACE_CONFIG=:4096:8'
    require_file_contains "$launch" '-e TOKENIZERS_PARALLELISM=false'
    require_file_contains "$launch" "-v \"$release:/workspace/repo:ro\""
    require_file_contains "$launch" "-v \"$run_dir:/workspace/run\""
    require_file_contains "$launch" "-v \"$dataset_root:/workspace/data:ro\""
    require_file_contains "$launch" '--entrypoint /usr/bin/python3'
    require_file_contains "$launch" '-I -c'
    require_file_contains "$launch" "$TRAIN_ISOLATED_BOOTSTRAP"
    require_file_contains "$launch" "\"$current_image_id\" -I -c"
    require_file_contains "$launch" "--config \"/workspace/repo/$config_path\""

    recorded_launch_sha="$(metadata_value "$metadata" launch_sha256)" ||
        fail invalid_run_metadata 'original launch digest is required for resume'
    require_sha256 launch_sha256 "$recorded_launch_sha"
    [[ "$(sha256sum "$launch" | awk '{print $1}')" == "$recorded_launch_sha" ]] ||
        fail run_identity_mismatch 'original launch script digest differs from run metadata'
}

assert_workspace_marker_or_absent() {
    local workspace="$1"
    if [[ -e "$workspace" && ! -d "$workspace" ]]; then
        fail foreign_workspace "'$workspace' exists but is not a directory"
    fi
    [[ ! -L "$workspace/.pseudo-brain-workspace-v2" ]] ||
        fail foreign_workspace 'workspace ownership marker cannot be a symbolic link'
    if [[ -f "$workspace/.pseudo-brain-workspace-v2" ]]; then
        grep -Fx 'pseudo-brain-workspace-v2' "$workspace/.pseudo-brain-workspace-v2" >/dev/null ||
            fail foreign_workspace 'workspace ownership marker has invalid content'
    fi
    if [[ -e "$workspace" && ! -f "$workspace/.pseudo-brain-workspace-v2" ]]; then
        if find "$workspace" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
            fail foreign_workspace "'$workspace' is nonempty and has no Pseudo-Brain v2 ownership marker"
        fi
    fi
}

assert_owned_workspace() {
    local workspace="$1"
    [[ -f "$workspace/.pseudo-brain-workspace-v2" && ! -L "$workspace/.pseudo-brain-workspace-v2" ]] ||
        fail unowned_workspace "'$workspace' has not been initialized by the Pseudo-Brain v2 sync tool"
    grep -Fx 'pseudo-brain-workspace-v2' "$workspace/.pseudo-brain-workspace-v2" >/dev/null ||
        fail unowned_workspace "'$workspace' has an invalid Pseudo-Brain v2 ownership marker"
}

assert_owned_historical_workspace() {
    local workspace="$1"
    [[ -f "$workspace/.irene-brain-workspace-v1" && ! -L "$workspace/.irene-brain-workspace-v1" ]] ||
        fail unowned_workspace "historical Irene workspace '$workspace' has no safe v1 marker"
    grep -Fx 'irene-brain-workspace-v1' "$workspace/.irene-brain-workspace-v1" >/dev/null ||
        fail unowned_workspace "historical Irene workspace '$workspace' has an invalid v1 marker"
}

release_dir() {
    local workspace="$1"
    local release_id="$2"
    require_slug release_id "$release_id"
    printf '%s/releases/%s\n' "$workspace" "$release_id"
}

container_smoke() {
    local workspace="$1"
    local release_id="$2"
    local image="$3"
    local config_path="$4"
    local cpu_limit="$5"
    local memory_limit_gib="$6"
    local image_id="$7"
    local release run_stamp smoke_run log_path container_script config_sha receipt_tmp receipt

    release="$(release_dir "$workspace" "$release_id")"
    [[ -d "$release" && ! -L "$release" ]] || fail missing_release "release '$release_id' does not exist or is unsafe"
    assert_clean_release_python_source "$release"
    require_relative_path config_path "$config_path"
    [[ -f "$release/$config_path" && ! -L "$release/$config_path" ]] || fail missing_config "config '$config_path' is missing or unsafe"
    [[ -f "$release/brain/src/irene_brain/training/train.py" && ! -L "$release/brain/src/irene_brain/training/train.py" ]] ||
        fail missing_trainer 'irene_brain.training.train module is missing or unsafe'
    require_uint cpu_limit "$cpu_limit"
    require_uint container_memory_gib "$memory_limit_gib"
    (( cpu_limit >= 1 && memory_limit_gib >= 4 )) || fail invalid_limit 'smoke CPU and memory limits are too small'

    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    smoke_run="$workspace/runs/smoke-${release_id}-${run_stamp}"
    mkdir -- "$smoke_run"
    log_path="$workspace/logs/smoke-${release_id}-${run_stamp}.log"
    require_image_id "$image_id"
    config_sha="$(sha256sum "$release/$config_path" | awk '{print $1}')"

    container_script="set -euo pipefail
python3 -I - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable inside the selected remote container'
device = torch.device('cuda')
x = torch.randn((64, 64), device=device, dtype=torch.bfloat16, requires_grad=True)
loss = (x @ x.transpose(0, 1)).float().square().mean()
loss.backward()
torch.cuda.synchronize()
print(f'torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)} bf16_backward=ok')
PY
cd /workspace/repo/brain
for test_path in tests/test_*.py; do
    test_module=\"tests.\$(basename \"\$test_path\" .py)\"
    python3 -I -c '$UNITTEST_ISOLATED_BOOTSTRAP' \"\$test_module\"
done
python3 -I -c '$TRAIN_ISOLATED_BOOTSTRAP' --config /workspace/repo/$config_path"

    set +e
    timeout --signal=TERM --kill-after=15s 300s docker run --rm \
        --pull never \
        --name "pseudo-brain-smoke-${release_id}" \
        --gpus all \
        --network none \
        --ipc host \
        --cpus "$cpu_limit" \
        --memory "${memory_limit_gib}g" \
        --pids-limit 512 \
        --stop-timeout 10 \
        --user "$(id -u):$(id -g)" \
        -e HOME=/workspace/run \
        -e PATH=/usr/sbin:/usr/bin \
        -e BASH_ENV= \
        -e ENV= \
        -e PSEUDO_BRAIN_RUN_DIR=/workspace/run \
        -e PSEUDO_BRAIN_MAX_STEPS=3 \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -e TOKENIZERS_PARALLELISM=false \
        -v "$release:/workspace/repo:ro" \
        -v "$smoke_run:/workspace/run" \
        --entrypoint /bin/bash \
        "$image_id" --noprofile --norc -c "$container_script" 2>&1 | tee "$log_path"
    local smoke_status=${PIPESTATUS[0]}
    set -e
    (( smoke_status == 0 )) || fail smoke_failed "foreground smoke exited with status $smoke_status; see $log_path"

    receipt="$workspace/smoke-receipts/${release_id}.env"
    receipt_tmp="${receipt}.tmp.$$"
    {
        printf 'workspace_contract=pseudo-brain-workspace-v2\n'
        printf 'release_id=%s\n' "$release_id"
        printf 'container_image=%s\n' "$image"
        printf 'container_image_id=%s\n' "$image_id"
        printf 'config_sha256=%s\n' "$config_sha"
        printf 'completed_utc=%s\n' "$run_stamp"
        printf 'log_path=%s\n' "$log_path"
    } > "$receipt_tmp"
    chmod 600 "$receipt_tmp"
    mv -- "$receipt_tmp" "$receipt"
    printf 'smoke_receipt=%s\n' "$receipt"
}

write_training_job() {
    local workspace="$1"
    local release_id="$2"
    local image="$3"
    local config_path="$4"
    local run_id="$5"
    local cpu_limit="$6"
    local memory_limit_gib="$7"
    local image_id="$8"
    local container_uid="$9"
    local container_gid="${10}"
    local release run_dir log_path job_path lock_path ready_path container_name

    release="$(release_dir "$workspace" "$release_id")"
    run_dir="$workspace/runs/$run_id"
    log_path="$workspace/logs/${run_id}.log"
    job_path="$run_dir/launch.sh"
    lock_path="$run_dir/.training.lock"
    ready_path="$run_dir/launch.ready"
    container_name="pseudo-brain-${run_id}"
    require_image_id "$image_id"
    require_uint container_uid "$container_uid"
    require_uint container_gid "$container_gid"

    mkdir -- "$run_dir" || fail run_claim_failed "could not exclusively create run '$run_id'"
    (
        set -o noclobber
        exec 8> "$lock_path" || exit 75
    ) || fail lock_create_failed "could not exclusively create '$lock_path'"
    chmod 600 "$lock_path" || fail lock_permission_failed "could not protect '$lock_path'"
    (
        set -o noclobber
        exec 8> "$job_path" || exit 75
        cat >&8 <<EOF
#!/bin/bash -p
PATH='/usr/sbin:/usr/bin'
export PATH
unset BASH_ENV ENV PYTHONHOME PYTHONPATH CDPATH
set -euo pipefail
exec 9>> "$run_dir/.training.lock" || { printf '%s\n' 'could not open training lock' >&2; exit 72; }
flock -n 9 || { printf '%s\n' 'training lock is already held' >&2; exit 73; }
(set -o noclobber; : > "$log_path") || { printf '%s\n' 'training log already exists' >&2; exit 74; }
(set -o noclobber; printf '%s\n' 'lock_acquired=1' > "$ready_path") || { printf '%s\n' 'launch handshake already exists' >&2; exit 75; }
docker run --rm \\
  --pull never \\
  --name "$container_name" \\
  --gpus all \\
  --network none \\
  --ipc host \\
  --cpus "$cpu_limit" \\
  --memory "${memory_limit_gib}g" \\
  --pids-limit 2048 \\
  --stop-timeout 30 \\
  --user "$container_uid:$container_gid" \\
  -e HOME=/workspace/run \\
  -e PATH=/usr/sbin:/usr/bin \\
  -e PSEUDO_BRAIN_RUN_DIR=/workspace/run \\
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
  -e TOKENIZERS_PARALLELISM=false \\
  -v "$release:/workspace/repo:ro" \\
  -v "$run_dir:/workspace/run" \\
  -v "$workspace/datasets:/workspace/data:ro" \\
  --entrypoint /usr/bin/python3 \\
  "$image_id" -I -c '$TRAIN_ISOLATED_BOOTSTRAP' --config "/workspace/repo/$config_path" \\
  2>&1 | tee -a "$log_path"
EOF
    ) || fail launch_create_failed "could not exclusively create '$job_path'"
    [[ -f "$job_path" && ! -L "$job_path" ]] || fail unsafe_launch_file "'$job_path' is not a regular owned launch file"
    chmod 700 "$job_path" || fail launch_permission_failed "could not protect '$job_path'"
}

write_rcq_staging_canary_job() {
    local workspace="$1"
    local release_id="$2"
    local image="$3"
    local run_id="$4"
    local cpu_limit="$5"
    local memory_limit_gib="$6"
    local image_id="$7"
    local container_uid="$8"
    local container_gid="$9"
    local config_path="$RCQ_STAGING_CANARY_CONFIG"
    local release run_dir log_path job_path lock_path ready_path container_name

    release="$(release_dir "$workspace" "$release_id")"
    run_dir="$workspace/runs/$run_id"
    log_path="$workspace/logs/${run_id}.log"
    job_path="$run_dir/launch.sh"
    lock_path="$run_dir/.training.lock"
    ready_path="$run_dir/launch.ready"
    container_name="pseudo-brain-${run_id}"
    require_image_id "$image_id"
    require_uint container_uid "$container_uid"
    require_uint container_gid "$container_gid"

    mkdir -- "$run_dir" || fail run_claim_failed "could not exclusively create run '$run_id'"
    (
        set -o noclobber
        exec 8> "$lock_path" || exit 75
    ) || fail lock_create_failed "could not exclusively create '$lock_path'"
    chmod 600 "$lock_path" || fail lock_permission_failed "could not protect '$lock_path'"
    (
        set -o noclobber
        exec 8> "$job_path" || exit 75
        cat >&8 <<EOF
#!/bin/bash -p
PATH='/usr/sbin:/usr/bin'
export PATH
unset BASH_ENV ENV PYTHONHOME PYTHONPATH CDPATH
set -euo pipefail
exec 9>> "$run_dir/.training.lock" || { printf '%s\n' 'could not open training lock' >&2; exit 72; }
flock -n 9 || { printf '%s\n' 'training lock is already held' >&2; exit 73; }
(set -o noclobber; : > "$log_path") || { printf '%s\n' 'training log already exists' >&2; exit 74; }
(set -o noclobber; printf '%s\n' 'lock_acquired=1' > "$ready_path") || { printf '%s\n' 'launch handshake already exists' >&2; exit 75; }
printf '%s\n' 'rcq_staging_canary_phase=initial stop_after_step=1' | tee -a "$log_path"
docker run --rm \
  --pull never \
  --name "$container_name" \
  --gpus all \
  --network none \
  --ipc host \
  --cpus "$cpu_limit" \
  --memory "${memory_limit_gib}g" \
  --pids-limit 512 \
  --stop-timeout 15 \
  --user "$container_uid:$container_gid" \
  -e HOME=/workspace/run \
  -e PATH=/usr/sbin:/usr/bin \
  -e PSEUDO_BRAIN_RUN_DIR=/workspace/run \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$release:/workspace/repo:ro" \
  -v "$run_dir:/workspace/run" \
  -v "$workspace/datasets:/workspace/data:ro" \
  --entrypoint /usr/bin/python3 \
  "$image_id" -I -c '$TRAIN_ISOLATED_BOOTSTRAP' \
  --config "/workspace/repo/$config_path" \
  --stop-after-step 1 \
  2>&1 | tee -a "$log_path"
checkpoint_path="$run_dir/checkpoints/step-00000001.pt"
[[ -f "\$checkpoint_path" && ! -L "\$checkpoint_path" ]] || { printf '%s\n' 'canary step-1 checkpoint is missing or linked' >&2; exit 76; }
checkpoint_sha="\$(sha256sum "\$checkpoint_path" | awk '{print \$1}')"
[[ "\$checkpoint_sha" =~ ^[a-f0-9]{64}\$ ]] || { printf '%s\n' 'canary step-1 checkpoint digest is invalid' >&2; exit 77; }
printf '%s\n' "rcq_staging_canary_phase=resume checkpoint_schema=2 checkpoint_sha256=\$checkpoint_sha" | tee -a "$log_path"
docker run --rm \
  --pull never \
  --name "$container_name" \
  --gpus all \
  --network none \
  --ipc host \
  --cpus "$cpu_limit" \
  --memory "${memory_limit_gib}g" \
  --pids-limit 512 \
  --stop-timeout 15 \
  --user "$container_uid:$container_gid" \
  -e HOME=/workspace/run \
  -e PATH=/usr/sbin:/usr/bin \
  -e PSEUDO_BRAIN_RUN_DIR=/workspace/run \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$release:/workspace/repo:ro" \
  -v "$run_dir:/workspace/run" \
  -v "$workspace/datasets:/workspace/data:ro" \
  --entrypoint /usr/bin/python3 \
  "$image_id" -I -c '$TRAIN_ISOLATED_BOOTSTRAP' \
  --config "/workspace/repo/$config_path" \
  --resume "/workspace/run/checkpoints/step-00000001.pt" \
  --checkpoint-sha256 "\$checkpoint_sha" \
  2>&1 | tee -a "$log_path"
python3 -I - "$run_dir" <<'PY_CANARY'
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def registered(path: Path):
    encoded = path.read_bytes()
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ValueError(f"{path.name} is not one newline-terminated object")
    value = json.loads(
        encoded[:-1].decode("utf-8", errors="strict"),
        object_pairs_hook=strict_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {token}")
        ),
    )
    canonical = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if encoded != canonical:
        raise ValueError(f"{path.name} is not byte-canonical")
    return value


run = Path(sys.argv[1])
checkpoint_one = run / "checkpoints" / "step-00000001.pt"
checkpoint_two = run / "checkpoints" / "step-00000002.pt"
for checkpoint in (checkpoint_one, checkpoint_two):
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError(f"unsafe or missing checkpoint: {checkpoint.name}")
pointer = registered(run / "checkpoints" / "latest.json")
required_pointer = {
    "schema_version", "checkpoint", "checkpoint_sha256", "optimizer_step"
}
if not isinstance(pointer, dict) or set(pointer) != required_pointer:
    raise ValueError("canary latest pointer has incompatible fields")
if (
    type(pointer["schema_version"]) is not int
    or pointer["schema_version"] != 1
    or pointer["checkpoint"] != checkpoint_two.name
    or type(pointer["optimizer_step"]) is not int
    or pointer["optimizer_step"] != 2
    or pointer["checkpoint_sha256"] != sha256(checkpoint_two.read_bytes()).hexdigest()
):
    raise ValueError("canary latest pointer does not bind the final checkpoint")

transition = registered(run / "stage-transition-01.json")
if (
    type(transition.get("schema_version")) is not int
    or transition.get("schema_version") != 1
    or type(transition.get("global_optimizer_step")) is not int
    or transition.get("global_optimizer_step") != 1
    or type(transition.get("from_stage_index")) is not int
    or transition.get("from_stage_index") != 0
    or type(transition.get("to_stage_index")) is not int
    or transition.get("to_stage_index") != 1
):
    raise ValueError("canary stage-transition artifact is invalid")
stage_state = transition.get("stage_state")
if (
    not isinstance(stage_state, dict)
    or type(stage_state.get("stage_index")) is not int
    or stage_state.get("stage_index") != 1
    or type(stage_state.get("stage_local_optimizer_step")) is not int
    or stage_state.get("stage_local_optimizer_step") != 0
    or stage_state.get("trainable_parameter_names")
    != ["model.value_per_thought.bias", "model.value_per_thought.weight"]
):
    raise ValueError("canary stage-transition state does not prove the freeze mask")

for step in (1, 2):
    invariant = registered(run / f"invariance-step-{step:08d}.json")
    report = invariant.get("report")
    if (
        type(invariant.get("schema_version")) is not int
        or invariant.get("schema_version") != 1
        or type(invariant.get("global_optimizer_step")) is not int
        or invariant.get("global_optimizer_step") != step
        or not isinstance(report, dict)
        or report.get("audit") != "nonvalue_action_state_v1"
        or report.get("passed") is not True
        or type(report.get("stage_index")) is not int
        or report.get("stage_index") != 1
    ):
        raise ValueError(f"canary invariance artifact at step {step} is invalid")

metric_lines = (run / "metrics.jsonl").read_bytes().splitlines()
if len(metric_lines) != 4:
    raise ValueError("canary metrics do not have the exact two-phase record count")
metric_keys = []
for encoded in metric_lines:
    record = json.loads(
        encoded.decode("utf-8", errors="strict"),
        object_pairs_hook=strict_object,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {token}")
        ),
    )
    metric_keys.append((record.get("split"), record.get("step")))
if metric_keys != [
    ("train", 1),
    ("validation", 1),
    ("train", 2),
    ("validation", 2),
]:
    raise ValueError("canary metrics do not prove exact prefix resume")
print("rcq_staging_canary=passed checkpoint_schema=2 resumed_from_step=1 final_step=2")
PY_CANARY
EOF
    ) || fail launch_create_failed "could not exclusively create '$job_path'"
    [[ -f "$job_path" && ! -L "$job_path" ]] || fail unsafe_launch_file "'$job_path' is not a regular owned launch file"
    chmod 700 "$job_path" || fail launch_permission_failed "could not protect '$job_path'"
}

write_resume_job() {
    local workspace="$1"
    local release_id="$2"
    local image="$3"
    local config_path="$4"
    local run_id="$5"
    local cpu_limit="$6"
    local memory_limit_gib="$7"
    local checkpoint_name="$8"
    local checkpoint_sha="$9"
    local attempt_id="${10}"
    local image_id="${11}"
    local metrics_bytes="${12}"
    local metrics_lines="${13}"
    local container_uid="${14}"
    local container_gid="${15}"
    local release run_dir attempt_dir log_path job_path ready_path container_name checkpoint_step_number

    release="$(release_dir "$workspace" "$release_id")"
    run_dir="$workspace/runs/$run_id"
    attempt_dir="$run_dir/resume-attempts/$attempt_id"
    log_path="$attempt_dir/train.log"
    job_path="$attempt_dir/launch.sh"
    ready_path="$attempt_dir/launch.ready"
    container_name="pseudo-brain-${run_id}"
    require_image_id "$image_id"
    require_uint metrics_bytes "$metrics_bytes"
    require_uint metrics_lines "$metrics_lines"
    require_uint container_uid "$container_uid"
    require_uint container_gid "$container_gid"
    [[ "$checkpoint_name" =~ ^step-[0-9]{8}\.pt$ ]] || fail invalid_checkpoint_name 'resume writer received an invalid checkpoint name'
    checkpoint_step_number="${checkpoint_name#step-}"
    checkpoint_step_number="${checkpoint_step_number%.pt}"
    checkpoint_step_number=$((10#$checkpoint_step_number))

    mkdir -- "$attempt_dir" || fail resume_attempt_exists "resume attempt '$attempt_id' already exists; refusing overwrite"
    (
        set -o noclobber
        exec 8> "$job_path" || exit 75
        cat >&8 <<EOF
#!/bin/bash -p
PATH='/usr/sbin:/usr/bin'
export PATH
unset BASH_ENV ENV PYTHONHOME PYTHONPATH CDPATH
set -euo pipefail
exec 9>> "$run_dir/.training.lock" || { printf '%s\n' 'could not open training lock' >&2; exit 72; }
flock -n 9 || { printf '%s\n' 'training lock is already held' >&2; exit 73; }
(set -o noclobber; : > "$log_path") || { printf '%s\n' 'resume log already exists' >&2; exit 74; }
printf '%s\n' 'resume_boundary checkpoint_step=$checkpoint_step_number metrics_bytes=$metrics_bytes metrics_lines=$metrics_lines' >> "$log_path" || { printf '%s\n' 'could not record resume boundary' >&2; exit 76; }
(set -o noclobber; printf '%s\n' 'lock_acquired=1' > "$ready_path") || { printf '%s\n' 'launch handshake already exists' >&2; exit 75; }
docker run --rm \\
  --pull never \\
  --name "$container_name" \\
  --gpus all \\
  --network none \\
  --ipc host \\
  --cpus "$cpu_limit" \\
  --memory "${memory_limit_gib}g" \\
  --pids-limit 2048 \\
  --stop-timeout 30 \\
  --user "$container_uid:$container_gid" \\
  -e HOME=/workspace/run \\
  -e PATH=/usr/sbin:/usr/bin \\
  -e PSEUDO_BRAIN_RUN_DIR=/workspace/run \\
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
  -e TOKENIZERS_PARALLELISM=false \\
  -v "$release:/workspace/repo:ro" \\
  -v "$run_dir:/workspace/run" \\
  -v "$workspace/datasets:/workspace/data:ro" \\
  --entrypoint /usr/bin/python3 \\
  "$image_id" -I -c '$TRAIN_ISOLATED_BOOTSTRAP' \\
  --config "/workspace/repo/$config_path" \\
  --resume "/workspace/run/checkpoints/$checkpoint_name" \\
  --checkpoint-sha256 "$checkpoint_sha" \\
  2>&1 | tee -a "$log_path"
EOF
    ) || fail launch_create_failed "could not exclusively create '$job_path'"
    [[ -f "$job_path" && ! -L "$job_path" ]] || fail unsafe_launch_file "'$job_path' is not a regular owned launch file"
    chmod 700 "$job_path" || fail launch_permission_failed "could not protect '$job_path'"
}

write_rcq_v2_pretraining_pin() {
    local workspace="$1"
    local release_id="$2"
    local expected_archive_sha="$3"
    local expected_registration_sha="$4"
    local image_reference="$5"
    local expected_image_id="$6"
    local release_root release archive_receipt registration config claim_root range_claim_id claim_leaf
    local pin_parent pin_dir pin_path campaign_artifact
    local actual_archive_sha actual_registration_sha actual_config_sha marker_sha created_utc pin_file_sha

    require_slug release_id "$release_id"
    require_sha256 release_archive_sha256 "$expected_archive_sha"
    require_sha256 registration_sha256 "$expected_registration_sha"
    require_image_reference "$image_reference"
    require_image_id "$expected_image_id"
    require_command python3
    require_command sha256sum

    release_root="$workspace/releases"
    assert_account_owned_directory "$release_root" "$workspace" 'release root'
    release_root="$DGX_SAFE_PATH"
    release="$release_root/$release_id"
    assert_account_owned_tree "$release" "$release_root" 'pretraining release tree'
    release="$DGX_SAFE_PATH"
    assert_clean_release_python_source "$release"

    archive_receipt="$release/.source-archive.sha256"
    assert_account_owned_regular_file "$archive_receipt" "$release" 'release archive receipt'
    archive_receipt="$DGX_SAFE_PATH"
    [[ "$(wc -c < "$archive_receipt")" == 65 && "$(wc -l < "$archive_receipt")" == 1 ]] ||
        fail invalid_release_receipt 'release archive receipt is not one lowercase SHA-256 line'
    actual_archive_sha="$(sed -n '1p' "$archive_receipt")"
    require_sha256 release_archive_sha256 "$actual_archive_sha"
    [[ "$actual_archive_sha" == "$expected_archive_sha" ]] ||
        fail release_archive_hash_mismatch 'installed release archive differs from the external pretraining pin'

    registration="$release/$RCQ_REGISTRATION_RELATIVE_PATH"
    assert_canonical_rcq_v2_registration_file "$registration" "$release"
    registration="$DGX_SAFE_PATH"
    actual_registration_sha="$(sha256sum "$registration" | awk '{print $1}')" ||
        fail registration_probe_failed 'could not hash the fixed RCQ-v2 registration'
    [[ "$actual_registration_sha" == "$expected_registration_sha" ]] ||
        fail registration_hash_mismatch 'installed registration differs from the external pretraining pin'

    config="$release/$RCQ_REFERENCE_CONFIG"
    assert_account_owned_regular_file "$config" "$release" 'fixed RCQ-v2 training configuration'
    config="$DGX_SAFE_PATH"
    actual_config_sha="$(sha256sum "$config" | awk '{print $1}')" ||
        fail config_hash_failed 'could not hash the fixed RCQ-v2 training configuration'

    claim_root="$workspace/final-claims"
    assert_account_owned_directory "$claim_root" "$workspace" 'persistent final-claim registry' 700
    claim_root="$DGX_SAFE_PATH"
    assert_account_owned_tree "$claim_root" "$workspace" 'persistent final-claim registry tree'
    claim_root="$DGX_SAFE_PATH"
    range_claim_id="$(python3 -I - "$registration" <<'PY'
import json
import re
import sys

value = json.load(open(sys.argv[1], "r", encoding="utf-8"))
receipt = value.get("receipt_directory") if isinstance(value, dict) else None
if not isinstance(receipt, str) or re.fullmatch(r"final-claims/[a-f0-9]{64}", receipt) is None:
    raise ValueError("registration has no fixed range claim")
print(receipt.split("/", 1)[1])
PY
)" || fail invalid_registration 'could not derive the fixed range-claim identity'
    require_sha256 range_claim_id "$range_claim_id"
    claim_leaf="$claim_root/$range_claim_id"
    [[ ! -e "$claim_leaf" && ! -L "$claim_leaf" ]] ||
        fail range_already_claimed 'the registered final TEST range is already retired in this workspace'
    for campaign_artifact in \
        "$workspace/runs/$RCQ_REFERENCE_RUN_ID" \
        "$workspace/logs/${RCQ_REFERENCE_RUN_ID}.log" \
        "$workspace/smoke-receipts/${release_id}.env" \
        "$workspace/rcq-staging-canary-receipts/${release_id}.env" \
        "$claim_root/$RCQ_READINESS_RELATIVE_PATH"; do
        [[ ! -e "$campaign_artifact" && ! -L "$campaign_artifact" ]] ||
            fail pretraining_chronology_violation "qualification artifact already exists before the pretraining pin: $campaign_artifact"
    done

    check_container_image "$image_reference"
    [[ "$DGX_IMAGE_ID" == "$expected_image_id" ]] ||
        fail image_id_mismatch 'cached container image differs from the externally approved immutable image ID'

    pin_parent="$workspace/qualification-pins"
    assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
    pin_parent="$DGX_SAFE_PATH"
    pin_dir="$pin_parent/rcq-v2-reference-v2"
    ensure_private_directory "$pin_dir" "$pin_parent" 'RCQ-v2 qualification pin directory'
    pin_dir="$DGX_SAFE_PATH"
    if find "$pin_dir" -mindepth 1 -print -quit | grep -q .; then
        fail pretraining_pin_exists 'qualification pin directory is not empty; pretraining trust root is create-only'
    fi
    pin_path="$pin_dir/$RCQ_PRETRAINING_PIN_FILENAME"
    marker_sha="$(sha256sum "$workspace/.pseudo-brain-workspace-v2" | awk '{print $1}')" ||
        fail marker_probe_failed 'could not hash the canonical workspace marker'
    [[ "$marker_sha" == "$RCQ_WORKSPACE_MARKER_SHA256" ]] ||
        fail unowned_workspace 'canonical workspace marker changed before pin publication'
    created_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)" || fail clock_failed 'could not record pretraining pin time'

    python3 -I - \
        "$registration" "$pin_dir" "$RCQ_PRETRAINING_PIN_FILENAME" \
        "$release_id" "$expected_archive_sha" "$expected_registration_sha" \
        "$actual_config_sha" "$image_reference" "$expected_image_id" \
        "$marker_sha" "$created_utc" <<'PY' || fail pretraining_pin_publish_failed 'could not publish the create-only RCQ-v2 pretraining pin'
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


def canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def require_hash(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"{name} is not one lowercase SHA-256")
    return value


def require_relative(value: object, expected: str, name: str) -> str:
    if value != expected or PurePosixPath(expected).is_absolute() or ".." in PurePosixPath(expected).parts:
        raise ValueError(f"{name} is not the fixed safe relative path")
    return expected


registration_path = Path(sys.argv[1])
pin_dir = Path(sys.argv[2])
pin_filename = sys.argv[3]
release_id = sys.argv[4]
archive_sha = require_hash(sys.argv[5], "archive_sha256")
registration_sha = require_hash(sys.argv[6], "registration_sha256")
observed_config_sha = require_hash(sys.argv[7], "observed_config_sha256")
image_reference = sys.argv[8]
image_id = sys.argv[9]
marker_sha = require_hash(sys.argv[10], "marker_file_sha256")
created_utc = sys.argv[11]
if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,62}", release_id) is None:
    raise ValueError("release id is unsafe")
if re.fullmatch(r"sha256:[a-f0-9]{64}", image_id) is None:
    raise ValueError("image id is not immutable")
if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created_utc) is None:
    raise ValueError("created_utc is not strict RFC3339 UTC")

registration = json.loads(registration_path.read_text(encoding="utf-8"))
protocol = registration.get("workspace_protocol")
expected_protocol = {
    "contract": "pseudo-brain-workspace-v2",
    "host_account_home_relative_path": "projects/pseudo-brain",
    "host_marker_relative_path": ".pseudo-brain-workspace-v2",
    "host_marker_exact_utf8": "pseudo-brain-workspace-v2\n",
    "dedicated_dispatcher_action": "rcq_v2_final_once_v1",
    "preclaim_dispatcher_action": "rcq_v2_preclaim_v1",
    "receipt_verifier_dispatcher_action": "rcq_v2_verify_receipt_v1",
    "container_release_root": "/workspace/repo",
    "container_run_root": "/workspace/run",
    "container_claim_registry_root": "/workspace/final-claims",
    "registration_release_relative_path": "registrations/rcq-v2-reference-v2.json",
    "readiness_receipt_relative_path": "preclaim-readiness/rcq-v2-reference-v2.json",
    "host_pin_directory_relative_path": "qualification-pins/rcq-v2-reference-v2",
    "container_pin_root": "/workspace/pins",
    "pretraining_pin_filename": "pretraining.json",
    "final_authorization_filename": "final-authorization.json",
}
if protocol != expected_protocol:
    raise ValueError("registration workspace protocol differs from the pinned host/container contract")
if registration.get("qualification_id") != "rcq_v2_reference_v2":
    raise ValueError("registration qualification id differs")
if registration.get("run_id") != "dgx-rcq-v2-reference-seed-1702":
    raise ValueError("registration run id differs")
if registration.get("run_seed") != 1702 or registration.get("final_step") != 2048:
    raise ValueError("registration run seed/final step differs")
if registration_path.read_bytes() and sha256(registration_path.read_bytes()).hexdigest() != registration_sha:
    raise ValueError("registration bytes changed")
config_raw_sha = require_hash(registration.get("config_raw_sha256"), "config_raw_sha256")
if config_raw_sha != observed_config_sha:
    raise ValueError("registered raw configuration hash differs from installed bytes")
config_canonical_sha = require_hash(
    registration.get("config_canonical_sha256"), "config_canonical_sha256"
)
source_tree_sha = require_hash(registration.get("source_tree_sha256"), "source_tree_sha256")
evaluator_bundle_sha = require_hash(
    registration.get("evaluator_bundle_sha256"), "evaluator_bundle_sha256"
)
batch_manifest_sha = require_hash(
    registration.get("batch_source_manifest_sha256"), "batch_source_manifest_sha256"
)
receipt_directory = registration.get("receipt_directory")
if type(receipt_directory) is not str or re.fullmatch(r"final-claims/[a-f0-9]{64}", receipt_directory) is None:
    raise ValueError("registration range-claim directory differs")
range_claim_id = receipt_directory.split("/", 1)[1]

payload: dict[str, object] = {
    "schema_version": 1,
    "action": "rcq_v2_pin_pretraining_v1",
    "qualification_id": "rcq_v2_reference_v2",
    "workspace": {
        "contract": "pseudo-brain-workspace-v2",
        "host_account_home_relative_path": "projects/pseudo-brain",
        "marker_relative_path": ".pseudo-brain-workspace-v2",
        "marker_file_sha256": marker_sha,
        "claim_registry_relative_path": "final-claims",
        "pin_directory_relative_path": "qualification-pins/rcq-v2-reference-v2",
    },
    "release": {
        "id": release_id,
        "relative_path": require_relative(
            f"releases/{release_id}", f"releases/{release_id}", "release.relative_path"
        ),
        "archive_sha256": archive_sha,
    },
    "registration": {
        "release_relative_path": "registrations/rcq-v2-reference-v2.json",
        "sha256": registration_sha,
    },
    "config": {
        "release_relative_path": "brain/configs/training/dgx-rcq-v2-reference.toml",
        "raw_sha256": config_raw_sha,
        "canonical_sha256": config_canonical_sha,
    },
    "source_tree_sha256": source_tree_sha,
    "evaluator_bundle_sha256": evaluator_bundle_sha,
    "batch_source_manifest_sha256": batch_manifest_sha,
    "runtime": {
        "container_image_reference": image_reference,
        "container_image_id": image_id,
        "container_release_root": "/workspace/repo",
        "container_run_root": "/workspace/run",
        "container_claim_registry_root": "/workspace/final-claims",
        "container_pin_root": "/workspace/pins",
        "network": "none",
    },
    "run": {
        "id": "dgx-rcq-v2-reference-seed-1702",
        "relative_path": "runs/dgx-rcq-v2-reference-seed-1702",
        "seed": 1702,
        "final_step": 2048,
    },
    "range_claim_id": range_claim_id,
    "created_utc": created_utc,
}
payload["pin_sha256"] = sha256(
    b"PSEUDOBRAINRCQPRETRAINPIN\x01" + canonical(payload).encode("utf-8")
).hexdigest()
encoded = (canonical(payload) + "\n").encode("utf-8")

if pin_filename != "pretraining.json":
    raise ValueError("pretraining filename differs")
directory_fd = os.open(pin_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    file_fd = os.open(
        pin_filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(file_fd, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(file_fd)
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY

    chmod 400 -- "$pin_path" || fail pretraining_pin_seal_failed 'could not seal the RCQ-v2 pretraining pin read-only'
    python3 -I - "$pin_path" "$pin_dir" <<'PY' || fail pretraining_pin_seal_failed 'could not fsync and verify the sealed RCQ-v2 pretraining pin'
import os
import stat
import sys

path = sys.argv[1]
parent = sys.argv[2]
fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
try:
    value = os.fstat(fd)
    if not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o400:
        raise ValueError("pretraining pin is not an exact 0400 regular file")
    if value.st_nlink != 1 or value.st_uid != os.geteuid():
        raise ValueError("pretraining pin ownership/link count changed")
    while os.read(fd, 1024 * 1024):
        pass
    os.fsync(fd)
finally:
    os.close(fd)
directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
    assert_account_owned_regular_file "$pin_path" "$pin_dir" 'RCQ-v2 pretraining pin'
    pin_path="$DGX_SAFE_PATH"
    [[ "$(stat -c '%a' -- "$pin_path")" == 400 ]] ||
        fail unsafe_artifact_mode 'RCQ-v2 pretraining pin must have mode 400'
    pin_file_sha="$(sha256sum "$pin_path" | awk '{print $1}')" ||
        fail pretraining_pin_probe_failed 'could not hash the published pretraining pin'
    require_sha256 pretraining_pin_sha256 "$pin_file_sha"
    printf 'pretraining_pin=%s\npretraining_pin_sha256=%s\n' "$pin_path" "$pin_file_sha"
}

open_rcq_v2_range_authority_lock() {
    local workspace="$1"
    local range_claim_id="$2"
    local claim_root lock_root
    require_sha256 range_claim_id "$range_claim_id"
    claim_root="$workspace/final-claims"
    assert_account_owned_directory "$claim_root" "$workspace" 'persistent final-claim registry' 700
    claim_root="$DGX_SAFE_PATH"
    ensure_private_directory "$claim_root/.locks" "$claim_root" 'final-claim host lock directory'
    lock_root="$DGX_SAFE_PATH"
    open_private_lock "$lock_root/${range_claim_id}.lock" "$lock_root" \
        'registered final-range host lock' 8
    RCQ_CLAIM_REGISTRY_PATH="$claim_root"
}

write_rcq_v2_final_authorization() {
    local workspace="$1"
    local expected_pretraining_file_sha="$2"
    local expected_latest_file_sha="$3"
    local expected_entry_checkpoint_sha="$4"
    local expected_checkpoint_sha="$5"
    local expected_readiness_file_sha="$6"
    local run_root run_dir checkpoint_root latest_path entry_checkpoint checkpoint readiness_path
    local final_path reviewed_utc final_file_sha observed_entries

    require_sha256 pretraining_pin_file_sha256 "$expected_pretraining_file_sha"
    require_sha256 latest_pointer_sha256 "$expected_latest_file_sha"
    require_sha256 entry_checkpoint_sha256 "$expected_entry_checkpoint_sha"
    require_sha256 checkpoint_sha256 "$expected_checkpoint_sha"
    require_sha256 readiness_sha256 "$expected_readiness_file_sha"
    [[ "$RCQ_PIN_FILE_SHA256" == "$expected_pretraining_file_sha" ]] ||
        fail pretraining_pin_mismatch 'externally reviewed pretraining pin differs from the canonical pin file'

    observed_entries="$(find "$RCQ_PIN_DIRECTORY" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)" ||
        fail pin_directory_probe_failed 'could not inspect the RCQ-v2 pin directory'
    [[ "$observed_entries" == "$RCQ_PRETRAINING_PIN_FILENAME" ]] ||
        fail final_authorization_exists 'qualification pin directory is not in the pre-authorization state'
    final_path="$RCQ_PIN_DIRECTORY/$RCQ_FINAL_AUTHORIZATION_FILENAME"

    run_root="$workspace/runs"
    assert_account_owned_directory "$run_root" "$workspace" 'run root'
    run_root="$DGX_SAFE_PATH"
    run_dir="$run_root/$RCQ_REFERENCE_RUN_ID"
    assert_account_owned_tree "$run_dir" "$run_root" 'fixed RCQ-v2 reference run'
    run_dir="$DGX_SAFE_PATH"
    assert_run_inactive "$RCQ_REFERENCE_RUN_ID"
    assert_run_lock_available "$run_dir"
    checkpoint_root="$run_dir/checkpoints"
    assert_account_owned_directory "$checkpoint_root" "$run_dir" 'fixed RCQ-v2 checkpoint directory'
    checkpoint_root="$DGX_SAFE_PATH"
    latest_path="$checkpoint_root/latest.json"
    entry_checkpoint="$checkpoint_root/$RCQ_ENTRY_CHECKPOINT_NAME"
    checkpoint="$checkpoint_root/$RCQ_FINAL_CHECKPOINT_NAME"
    assert_account_owned_regular_file "$latest_path" "$checkpoint_root" 'terminal latest pointer'
    latest_path="$DGX_SAFE_PATH"
    assert_account_owned_regular_file "$entry_checkpoint" "$checkpoint_root" 'retained entry checkpoint'
    entry_checkpoint="$DGX_SAFE_PATH"
    assert_account_owned_regular_file "$checkpoint" "$checkpoint_root" 'terminal checkpoint'
    checkpoint="$DGX_SAFE_PATH"
    readiness_path="$RCQ_CLAIM_REGISTRY_PATH/$RCQ_READINESS_RELATIVE_PATH"
    assert_account_owned_regular_file "$readiness_path" "$RCQ_CLAIM_REGISTRY_PATH" 'preclaim readiness receipt'
    readiness_path="$DGX_SAFE_PATH"
    reviewed_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)" || fail clock_failed 'could not record final authorization review time'

    python3 -I - \
        "$RCQ_PIN_PRETRAINING_PATH" "$RCQ_PIN_DIRECTORY" "$RCQ_FINAL_AUTHORIZATION_FILENAME" \
        "$latest_path" "$entry_checkpoint" "$checkpoint" "$readiness_path" \
        "$expected_pretraining_file_sha" "$expected_latest_file_sha" \
        "$expected_entry_checkpoint_sha" "$expected_checkpoint_sha" \
        "$expected_readiness_file_sha" "$RCQ_PIN_RANGE_CLAIM_ID" "$reviewed_utc" <<'PY' || fail final_authorization_publish_failed 'could not validate evidence and publish the create-only final authorization'
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def hash_string(value, name):
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"{name} is not one lowercase SHA-256")
    return value


def strict_json_line(path):
    encoded = path.read_bytes()
    if not encoded.endswith(b"\n") or encoded.endswith(b"\n\n"):
        raise ValueError(f"{path.name} is not one newline-terminated object")
    value = json.loads(
        encoded[:-1].decode("utf-8", errors="strict"),
        object_pairs_hook=strict_object,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if encoded != (canonical(value) + "\n").encode("utf-8"):
        raise ValueError(f"{path.name} is not byte-canonical")
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not an object")
    return value, encoded


def file_sha(path):
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


pretraining_path = Path(sys.argv[1])
pin_dir = Path(sys.argv[2])
filename = sys.argv[3]
latest_path = Path(sys.argv[4])
entry_path = Path(sys.argv[5])
checkpoint_path = Path(sys.argv[6])
readiness_path = Path(sys.argv[7])
expected_pretraining_file = hash_string(sys.argv[8], "pretraining file sha256")
expected_latest_file = hash_string(sys.argv[9], "latest file sha256")
expected_entry = hash_string(sys.argv[10], "entry checkpoint sha256")
expected_checkpoint = hash_string(sys.argv[11], "checkpoint sha256")
expected_readiness_file = hash_string(sys.argv[12], "readiness file sha256")
range_claim_id = hash_string(sys.argv[13], "range claim id")
reviewed_utc = sys.argv[14]
if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", reviewed_utc) is None:
    raise ValueError("reviewed_utc is not strict RFC3339 UTC")
if filename != "final-authorization.json":
    raise ValueError("final authorization filename differs")

pretraining, pretraining_encoded = strict_json_line(pretraining_path)
if sha256(pretraining_encoded).hexdigest() != expected_pretraining_file:
    raise ValueError("pretraining pin differs from external review")
pretraining_semantic = hash_string(pretraining.get("pin_sha256"), "pretraining pin sha256")
if pretraining.get("range_claim_id") != range_claim_id:
    raise ValueError("pretraining range claim differs")

latest, latest_encoded = strict_json_line(latest_path)
if sha256(latest_encoded).hexdigest() != expected_latest_file:
    raise ValueError("latest pointer differs from external review")
if set(latest) != {"schema_version", "checkpoint", "checkpoint_sha256", "optimizer_step"}:
    raise ValueError("latest pointer fields differ")
if latest != {
    "schema_version": 1,
    "checkpoint": "step-00002048.pt",
    "checkpoint_sha256": expected_checkpoint,
    "optimizer_step": 2048,
}:
    raise ValueError("latest pointer is not the fixed terminal checkpoint")
if file_sha(entry_path) != expected_entry:
    raise ValueError("entry checkpoint differs from external review")
if file_sha(checkpoint_path) != expected_checkpoint:
    raise ValueError("terminal checkpoint differs from external review")

readiness, readiness_encoded = strict_json_line(readiness_path)
if sha256(readiness_encoded).hexdigest() != expected_readiness_file:
    raise ValueError("readiness receipt differs from external review")
expected_readiness_fields = {
    "schema_version", "action", "qualification_id", "evaluator_id", "status",
    "pretraining_pin", "registration_sha256", "config_raw_sha256",
    "config_canonical_sha256", "source_tree_sha256", "evaluator_bundle_sha256",
    "batch_source_manifest_sha256", "entry_checkpoint", "checkpoint_sha256",
    "latest_file_sha256", "checkpoint_step", "checkpoint_stage_index",
    "runtime_fingerprint_sha256", "entry_gate_report_sha256",
    "completion_gate_report_sha256", "invariance_report_sha256", "metrics_prefix",
    "entry_metrics_prefix", "train_lookup_sha256", "entry_development_replay",
    "development_replay", "entry_terminal_nonvalue_identity", "workspace_protocol",
    "range_claim_id", "receipt_directory", "range_claim_registry_observed_empty",
    "sealed_test_datasets_constructed", "sealed_test_examples_opened",
    "readiness_sha256",
}
if set(readiness) != expected_readiness_fields:
    raise ValueError("readiness receipt fields differ")
if (
    readiness["schema_version"] != 1
    or readiness["action"] != "rcq_v2_preclaim_v1"
    or readiness["qualification_id"] != "rcq_v2_reference_v2"
    or readiness["status"] != "ready_for_once_only_final"
    or readiness["range_claim_id"] != range_claim_id
    or readiness["receipt_directory"] != f"final-claims/{range_claim_id}"
    or readiness["range_claim_registry_observed_empty"] is not True
    or readiness["sealed_test_datasets_constructed"] != 0
    or readiness["sealed_test_examples_opened"] != 0
    or readiness["checkpoint_sha256"] != expected_checkpoint
    or readiness["latest_file_sha256"] != expected_latest_file
    or readiness["checkpoint_step"] != 2048
):
    raise ValueError("readiness receipt identity differs")
readiness_pretraining = readiness["pretraining_pin"]
if readiness_pretraining != {
    "relative_path": "pretraining.json",
    "file_sha256": expected_pretraining_file,
    "pin_sha256": pretraining_semantic,
}:
    raise ValueError("readiness pretraining binding differs")
readiness_entry = readiness["entry_checkpoint"]
if readiness_entry != {
    "relative_path": "checkpoints/step-00001536.pt",
    "sha256": expected_entry,
    "optimizer_step": 1536,
    "stage_index": 1,
    "stage_local_optimizer_step": 0,
}:
    raise ValueError("readiness entry-checkpoint binding differs")
observed_readiness_semantic = hash_string(
    readiness["readiness_sha256"], "readiness semantic sha256"
)
readiness_body = dict(readiness)
readiness_body.pop("readiness_sha256")
expected_readiness_semantic = sha256(
    b"IRENERCQREADINESS\x01" + canonical(readiness_body).encode("utf-8")
).hexdigest()
if observed_readiness_semantic != expected_readiness_semantic:
    raise ValueError("readiness semantic digest differs")

body = {
    "schema_version": 1,
    "action": "rcq_v2_authorize_final_v1",
    "qualification_id": "rcq_v2_reference_v2",
    "pretraining": {
        "relative_path": "pretraining.json",
        "file_sha256": expected_pretraining_file,
        "pin_sha256": pretraining_semantic,
    },
    "latest": {
        "run_relative_path": "runs/dgx-rcq-v2-reference-seed-1702",
        "relative_path": "checkpoints/latest.json",
        "file_sha256": expected_latest_file,
        "checkpoint": "step-00002048.pt",
        "checkpoint_sha256": expected_checkpoint,
        "optimizer_step": 2048,
    },
    "entry_checkpoint": {
        "relative_path": "checkpoints/step-00001536.pt",
        "sha256": expected_entry,
    },
    "checkpoint": {
        "relative_path": "checkpoints/step-00002048.pt",
        "sha256": expected_checkpoint,
    },
    "readiness": {
        "relative_path": "final-claims/preclaim-readiness/rcq-v2-reference-v2.json",
        "file_sha256": expected_readiness_file,
        "readiness_sha256": observed_readiness_semantic,
    },
    "range_claim_id": range_claim_id,
    "reviewed_utc": reviewed_utc,
}
body["authorization_sha256"] = sha256(
    b"PSEUDOBRAINRCQFINALAUTH\x01" + canonical(body).encode("utf-8")
).hexdigest()
encoded = (canonical(body) + "\n").encode("utf-8")

directory_fd = os.open(pin_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
descriptor = -1
try:
    if set(os.listdir(directory_fd)) != {"pretraining.json"}:
        raise ValueError("pin directory changed before authorization publication")
    descriptor = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    with os.fdopen(descriptor, "wb", closefd=False) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.fchmod(descriptor, 0o400)
    os.fsync(descriptor)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
    ):
        raise ValueError("published final authorization ownership/mode differs")
    os.close(descriptor)
    descriptor = -1
    os.fsync(directory_fd)
finally:
    if descriptor >= 0:
        os.close(descriptor)
    os.close(directory_fd)
PY

    assert_account_owned_regular_file "$final_path" "$RCQ_PIN_DIRECTORY" 'RCQ-v2 final authorization'
    final_path="$DGX_SAFE_PATH"
    [[ "$(stat -c '%a' -- "$final_path")" == 400 ]] ||
        fail unsafe_artifact_mode 'RCQ-v2 final authorization must have exact mode 0400'
    final_file_sha="$(sha256sum "$final_path" | awk '{print $1}')" ||
        fail final_authorization_probe_failed 'could not hash the final authorization'
    require_sha256 final_authorization_file_sha256 "$final_file_sha"
    printf 'final_authorization=%s\nfinal_authorization_file_sha256=%s\n' \
        "$final_path" "$final_file_sha"
}

bind_rcq_v2_receipt_verification_authority() {
    local workspace="$1"
    local summary pin_parent pin_path release_root release archive_receipt registration config actual
    summary="$(load_rcq_v2_pretraining_pin_summary "$workspace")" ||
        fail invalid_pretraining_pin 'could not load the receipt-verification pretraining trust root'
    IFS=$'\t' read -r \
        RCQ_PIN_RELEASE_ID RCQ_PIN_ARCHIVE_SHA256 RCQ_PIN_REGISTRATION_SHA256 \
        RCQ_PIN_CONFIG_SHA256 RCQ_PIN_IMAGE_REFERENCE RCQ_PIN_IMAGE_ID \
        RCQ_PIN_RANGE_CLAIM_ID RCQ_PIN_SEMANTIC_SHA256 <<< "$summary"
    require_sha256 range_claim_id "$RCQ_PIN_RANGE_CLAIM_ID"
    require_sha256 pretraining_pin_sha256 "$RCQ_PIN_SEMANTIC_SHA256"
    pin_parent="$workspace/qualification-pins"
    assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
    pin_parent="$DGX_SAFE_PATH"
    RCQ_PIN_DIRECTORY="$pin_parent/rcq-v2-reference-v2"
    assert_account_owned_directory "$RCQ_PIN_DIRECTORY" "$pin_parent" 'RCQ-v2 qualification pin directory' 700
    RCQ_PIN_DIRECTORY="$DGX_SAFE_PATH"
    pin_path="$RCQ_PIN_DIRECTORY/$RCQ_PRETRAINING_PIN_FILENAME"
    assert_account_owned_regular_file "$pin_path" "$RCQ_PIN_DIRECTORY" 'RCQ-v2 pretraining pin'
    RCQ_PIN_PRETRAINING_PATH="$DGX_SAFE_PATH"
    [[ "$(stat -c '%a' -- "$RCQ_PIN_PRETRAINING_PATH")" == 400 ]] ||
        fail unsafe_artifact_mode 'RCQ-v2 pretraining pin must have exact mode 0400'
    RCQ_PIN_FILE_SHA256="$(sha256sum "$RCQ_PIN_PRETRAINING_PATH" | awk '{print $1}')" ||
        fail pretraining_pin_probe_failed 'could not hash the RCQ-v2 pretraining pin'
    require_sha256 pretraining_pin_file_sha256 "$RCQ_PIN_FILE_SHA256"

    release_root="$workspace/releases"
    assert_account_owned_directory "$release_root" "$workspace" 'release root'
    release_root="$DGX_SAFE_PATH"
    release="$release_root/$RCQ_PIN_RELEASE_ID"
    assert_account_owned_tree "$release" "$release_root" 'receipt-verification pinned release tree'
    release="$DGX_SAFE_PATH"
    assert_clean_release_python_source "$release"
    archive_receipt="$release/.source-archive.sha256"
    assert_account_owned_regular_file "$archive_receipt" "$release" 'release archive receipt'
    archive_receipt="$DGX_SAFE_PATH"
    actual="$(sed -n '1p' "$archive_receipt")"
    [[ "$(wc -c < "$archive_receipt")" == 65 && "$(wc -l < "$archive_receipt")" == 1 &&
        "$actual" == "$RCQ_PIN_ARCHIVE_SHA256" ]] ||
        fail release_archive_hash_mismatch 'receipt-verification release archive changed'
    registration="$release/$RCQ_REGISTRATION_RELATIVE_PATH"
    assert_canonical_rcq_v2_registration_file "$registration" "$release"
    registration="$DGX_SAFE_PATH"
    [[ "$(sha256sum "$registration" | awk '{print $1}')" == "$RCQ_PIN_REGISTRATION_SHA256" ]] ||
        fail registration_hash_mismatch 'receipt-verification registration changed'
    config="$release/$RCQ_REFERENCE_CONFIG"
    assert_account_owned_regular_file "$config" "$release" 'fixed RCQ-v2 training configuration'
    config="$DGX_SAFE_PATH"
    [[ "$(sha256sum "$config" | awk '{print $1}')" == "$RCQ_PIN_CONFIG_SHA256" ]] ||
        fail config_hash_mismatch 'receipt-verification training configuration changed'
    check_container_image "$RCQ_PIN_IMAGE_REFERENCE"
    [[ "$DGX_IMAGE_ID" == "$RCQ_PIN_IMAGE_ID" ]] ||
        fail image_id_mismatch 'receipt-verification image differs from the pretraining pin'
    RCQ_PIN_RELEASE_PATH="$release"
}

prepare_rcq_v2_evaluator_paths() {
    local workspace="$1"
    local operation="$2"
    local pin_entries run_root claim_leaf final_authorization readiness_path
    [[ "$operation" == preclaim || "$operation" == final-once || "$operation" == verify-receipt ]] ||
        fail invalid_evaluator_operation 'unsupported trusted evaluator operation'

    pin_entries="$(find "$RCQ_PIN_DIRECTORY" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)" ||
        fail pin_directory_probe_failed 'could not inspect the RCQ-v2 pin directory'
    if [[ "$operation" == preclaim ]]; then
        [[ "$pin_entries" == "$RCQ_PRETRAINING_PIN_FILENAME" ]] ||
            fail invalid_pin_state 'preclaim requires pretraining.json and no final authorization'
    else
        [[ "$pin_entries" == "$RCQ_FINAL_AUTHORIZATION_FILENAME"$'\n'"$RCQ_PRETRAINING_PIN_FILENAME" ]] ||
            fail invalid_pin_state 'final/verification requires exactly the two frozen qualification pins'
        final_authorization="$RCQ_PIN_DIRECTORY/$RCQ_FINAL_AUTHORIZATION_FILENAME"
        assert_account_owned_regular_file "$final_authorization" "$RCQ_PIN_DIRECTORY" 'RCQ-v2 final authorization'
        final_authorization="$DGX_SAFE_PATH"
        [[ "$(stat -c '%a' -- "$final_authorization")" == 400 ]] ||
            fail unsafe_artifact_mode 'RCQ-v2 final authorization must have exact mode 0400'
    fi

    run_root="$workspace/runs"
    assert_account_owned_directory "$run_root" "$workspace" 'run root'
    run_root="$DGX_SAFE_PATH"
    RCQ_EVALUATOR_RUN_PATH="$run_root/$RCQ_REFERENCE_RUN_ID"
    assert_account_owned_tree "$RCQ_EVALUATOR_RUN_PATH" "$run_root" 'fixed RCQ-v2 reference run'
    RCQ_EVALUATOR_RUN_PATH="$DGX_SAFE_PATH"
    assert_run_inactive "$RCQ_REFERENCE_RUN_ID"
    assert_run_lock_available "$RCQ_EVALUATOR_RUN_PATH"

    assert_account_owned_directory "$RCQ_CLAIM_REGISTRY_PATH" "$workspace" \
        'persistent final-claim registry' 700
    RCQ_CLAIM_REGISTRY_PATH="$DGX_SAFE_PATH"
    claim_leaf="$RCQ_CLAIM_REGISTRY_PATH/$RCQ_PIN_RANGE_CLAIM_ID"
    if [[ "$operation" == verify-receipt ]]; then
        if [[ -e "$claim_leaf" || -L "$claim_leaf" ]]; then
            assert_account_owned_tree "$claim_leaf" "$RCQ_CLAIM_REGISTRY_PATH" \
                'retired RCQ-v2 final range receipt tree'
            claim_leaf="$DGX_SAFE_PATH"
            assert_account_owned_directory "$claim_leaf" "$RCQ_CLAIM_REGISTRY_PATH" \
                'retired RCQ-v2 final range receipt directory' 700
        fi
    else
        [[ ! -e "$claim_leaf" && ! -L "$claim_leaf" ]] ||
            fail range_already_claimed 'the registered final TEST range is already retired in this workspace'
    fi
    readiness_path="$RCQ_CLAIM_REGISTRY_PATH/$RCQ_READINESS_RELATIVE_PATH"
    if [[ "$operation" != preclaim ]]; then
        assert_account_owned_regular_file "$readiness_path" "$RCQ_CLAIM_REGISTRY_PATH" \
            'preclaim readiness receipt'
    fi
}

run_rcq_v2_evaluator_container() {
    local operation="$1"
    local bootstrap timeout_seconds cpu_limit memory_limit_gib claim_mount_mode container_name
    local container_uid container_gid status
    local -a docker_arguments
    case "$operation" in
        preclaim)
            bootstrap="$RCQ_PRECLAIM_ISOLATED_BOOTSTRAP"
            timeout_seconds=3600
            cpu_limit=12
            memory_limit_gib=96
            claim_mount_mode=rw
            container_name=pseudo-brain-rcq-v2-preclaim
            ;;
        final-once)
            bootstrap="$RCQ_FINAL_ONCE_ISOLATED_BOOTSTRAP"
            timeout_seconds=21600
            cpu_limit=12
            memory_limit_gib=96
            claim_mount_mode=rw
            container_name=pseudo-brain-rcq-v2-final-once
            ;;
        verify-receipt)
            bootstrap="$RCQ_VERIFY_RECEIPT_ISOLATED_BOOTSTRAP"
            timeout_seconds=600
            cpu_limit=2
            memory_limit_gib=16
            claim_mount_mode=ro
            container_name=pseudo-brain-rcq-v2-verify-receipt
            ;;
        *) fail invalid_evaluator_operation 'unsupported trusted evaluator operation' ;;
    esac
    container_uid="$(id -u)" || fail identity_probe_failed 'could not read container UID'
    container_gid="$(id -g)" || fail identity_probe_failed 'could not read container GID'
    require_uint container_uid "$container_uid"
    require_uint container_gid "$container_gid"
    require_command docker
    require_command timeout
    containers="$(docker ps -a --filter "name=^/${container_name}$" --format '{{.Names}}')" ||
        fail container_probe_failed 'could not inspect trusted evaluator container state'
    [[ -z "$containers" ]] || fail run_active "trusted evaluator container still exists: $containers"
    if [[ "$operation" != verify-receipt ]]; then
        check_gpu_idle
    fi

    docker_arguments=(
        run --rm
        --pull never \
        --name "$container_name" \
        --network none \
        --ipc host \
        --cpus "$cpu_limit" \
        --memory "${memory_limit_gib}g" \
        --pids-limit 2048 \
        --stop-timeout 30 \
        --user "$container_uid:$container_gid" \
        -e HOME=/tmp \
        -e PATH=/usr/sbin:/usr/bin \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -e TOKENIZERS_PARALLELISM=false \
        -v "$RCQ_PIN_RELEASE_PATH:/workspace/repo:ro" \
        -v "$RCQ_EVALUATOR_RUN_PATH:/workspace/run:ro" \
        -v "$RCQ_CLAIM_REGISTRY_PATH:/workspace/final-claims:$claim_mount_mode" \
        -v "$RCQ_PIN_DIRECTORY:/workspace/pins:ro" \
        --entrypoint /usr/bin/python3 \
        "$RCQ_PIN_IMAGE_ID" -I -B -c "$bootstrap"
    )
    if [[ "$operation" != verify-receipt ]]; then
        docker_arguments=(run --rm --pull never --name "$container_name" --gpus all "${docker_arguments[@]:6}")
    fi
    set +e
    timeout --signal=TERM --kill-after=30s "${timeout_seconds}s" \
        docker "${docker_arguments[@]}"
    status=$?
    set -e
    return "$status"
}

action="${1:-}"
[[ -n "$action" ]] || fail missing_action 'no action was provided'
shift
assert_trusted_host_runtime

case "$action" in
    preflight)
        [[ $# -eq 4 ]] || fail invalid_arguments 'preflight expects workspace image min_disk_gib min_memory_gib'
        workspace="$(resolve_workspace "$1")"
        image="$2"
        min_disk_gib="$3"
        min_memory_gib="$4"
        assert_architecture
        assert_workspace_marker_or_absent "$workspace"
        check_disk_gib "$workspace" "$min_disk_gib"
        check_memory_gib "$min_memory_gib"
        check_gpu_idle
        require_command python3
        printf 'system_python=%s\n' "$(python3 --version 2>&1)"
        if python3 -I -c 'import torch' >/dev/null 2>&1; then
            python3 -I -c 'import torch; print(f"system_torch={torch.__version__}")'
        else
            printf 'system_torch=absent (allowed; container torch is authoritative)\n'
        fi
        check_container_image "$image"
        current_image_id="$DGX_IMAGE_ID"
        docker run --rm \
            --pull never \
            --gpus all \
            --network none \
            --cpus 1 \
            --memory 4g \
            --pids-limit 256 \
            --entrypoint /usr/bin/python3 \
            "$current_image_id" -I -c 'import platform, torch; assert platform.machine() in {"aarch64", "arm64"}; assert torch.cuda.is_available(); print(f"container_python={platform.python_version()} torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)} bf16={torch.cuda.is_bf16_supported()}")'
        ;;

    prepare_sync)
        [[ $# -eq 2 ]] || fail invalid_arguments 'prepare_sync expects workspace min_disk_gib'
        workspace="$(resolve_workspace "$1")"
        assert_architecture
        assert_workspace_marker_or_absent "$workspace"
        check_disk_gib "$workspace" "$2"
        mkdir -p -- "$workspace/incoming" "$workspace/releases" "$workspace/runs" \
            "$workspace/logs" "$workspace/checkpoints" "$workspace/datasets" \
            "$workspace/smoke-receipts" "$workspace/rcq-staging-canary-receipts"
        if [[ ! -f "$workspace/.pseudo-brain-workspace-v2" ]]; then
            (set -o noclobber; printf 'pseudo-brain-workspace-v2\n' > "$workspace/.pseudo-brain-workspace-v2") ||
                fail marker_race 'workspace ownership marker appeared concurrently'
        fi
        if [[ ! -e "$workspace/final-claims" && ! -L "$workspace/final-claims" ]]; then
            (umask 077; mkdir -- "$workspace/final-claims") ||
                fail authority_directory_create_failed 'could not create the persistent final-claim registry'
        fi
        if [[ ! -e "$workspace/qualification-pins" && ! -L "$workspace/qualification-pins" ]]; then
            (umask 077; mkdir -- "$workspace/qualification-pins") ||
                fail authority_directory_create_failed 'could not create the qualification pin root'
        fi
        assert_account_owned_directory "$workspace/final-claims" "$workspace" \
            'persistent final-claim registry' 700
        assert_account_owned_directory "$workspace/qualification-pins" "$workspace" \
            'qualification pin root' 700
        printf 'workspace_ready=%s\n' "$workspace"
        ;;

    rcq_v2_pin_pretraining_v1)
        [[ $# -eq 5 ]] || fail invalid_arguments \
            'rcq_v2_pin_pretraining_v1 expects release_id archive_sha256 registration_sha256 image_reference image_id'
        workspace="$(resolve_canonical_rcq_v2_workspace)"
        assert_architecture
        pin_parent="$workspace/qualification-pins"
        assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
        pin_parent="$DGX_SAFE_PATH"
        require_command flock
        open_private_lock "$pin_parent/.rcq-v2-reference-v2.lock" "$pin_parent" \
            'RCQ-v2 qualification authority lock'
        write_rcq_v2_pretraining_pin "$workspace" "$1" "$2" "$3" "$4" "$5"
        ;;

    install_sync)
        [[ $# -eq 4 ]] || fail invalid_arguments 'install_sync expects workspace archive_name release_id sha256'
        workspace="$(resolve_workspace "$1")"
        archive_name="$2"
        release_id="$3"
        expected_sha="$4"
        assert_owned_workspace "$workspace"
        require_command python3
        [[ "$archive_name" =~ ^pseudo-brain-[a-f0-9]{12}\.tgz$ ]] || fail invalid_archive 'archive name is invalid'
        require_slug release_id "$release_id"
        [[ "$expected_sha" =~ ^[a-f0-9]{64}$ ]] || fail invalid_sha 'archive sha256 is invalid'
        archive_path="$workspace/incoming/$archive_name"
        [[ -f "$archive_path" ]] || fail missing_archive "incoming archive '$archive_name' is missing"
        actual_sha="$(sha256sum "$archive_path" | awk '{print $1}')"
        [[ "$actual_sha" == "$expected_sha" ]] || fail archive_hash_mismatch 'incoming archive hash does not match'
        registration_members=0
        while IFS= read -r entry; do
            [[ ( "$entry" == brain/* || "$entry" == registrations/rcq-v2-reference-v2.json ) &&
                "/$entry/" != *"/../"* && "$entry" != /* ]] ||
                fail unsafe_archive "unsafe archive member '$entry'"
            if [[ "$entry" == registrations/rcq-v2-reference-v2.json ]]; then
                registration_members=$((registration_members + 1))
            fi
        done < <(tar -tzf "$archive_path")
        (( registration_members == 1 )) ||
            fail incomplete_release 'archive must contain the fixed RCQ-v2 registration exactly once'
        archive_listing="$(LC_ALL=C tar -tvzf "$archive_path")" ||
            fail unsafe_archive 'could not inspect archive entry types'
        while IFS= read -r listing; do
            [[ "${listing:0:1}" == - || "${listing:0:1}" == d ]] ||
                fail unsafe_archive "archive contains a link or special entry: $listing"
        done <<< "$archive_listing"
        destination="$workspace/releases/$release_id"
        if [[ -e "$destination" || -L "$destination" ]]; then
            [[ -d "$destination" && ! -L "$destination" ]] || fail release_collision 'release destination is not a safe directory'
            [[ -f "$destination/.source-archive.sha256" && ! -L "$destination/.source-archive.sha256" ]] || fail release_collision 'existing release has no safe source receipt'
            [[ "$(cat "$destination/.source-archive.sha256")" == "$actual_sha" ]] || fail release_collision 'release ID exists with different content'
            rm -- "$archive_path" || fail archive_cleanup_failed 'could not remove installed incoming archive'
            printf 'release_already_present=%s\n' "$destination"
            exit 0
        fi
        staging="$workspace/releases/.install-${release_id}-$$"
        trap 'rm -rf -- "$staging"' EXIT
        mkdir -- "$staging" || fail staging_create_failed 'could not exclusively create release staging directory'
        tar -xzf "$archive_path" --no-same-owner --no-same-permissions -C "$staging" || fail archive_extract_failed 'could not extract release archive'
        assert_safe_artifact_tree "$staging" "$staging" 'extracted release'
        hardlinked_entry="$(find "$staging" -type f -links +1 -print -quit)" ||
            fail unsafe_archive 'could not inspect extracted release link counts'
        [[ -z "$hardlinked_entry" ]] ||
            fail unsafe_archive "hardlinked files are forbidden in synced releases: $hardlinked_entry"
        [[ -f "$staging/brain/pyproject.toml" ]] || fail incomplete_release 'brain/pyproject.toml is missing'
        assert_clean_release_python_source "$staging"
        assert_canonical_rcq_v2_registration_file \
            "$staging/registrations/rcq-v2-reference-v2.json" "$staging"
        (set -o noclobber; printf '%s\n' "$actual_sha" > "$staging/.source-archive.sha256") ||
            fail source_receipt_exists 'release source receipt appeared concurrently'
        chmod -R a-w -- "$staging" || fail release_permission_failed 'could not make release tree read-only'
        mv -- "$staging" "$destination" || fail release_install_failed 'could not atomically install release'
        trap - EXIT
        rm -- "$archive_path" || fail archive_cleanup_failed 'could not remove installed incoming archive'
        printf 'release_id=%s\nrelease_path=%s\narchive_sha256=%s\n' "$release_id" "$destination" "$actual_sha"
        ;;

    smoke|rcq_v2_smoke_v1)
        if [[ "$action" == rcq_v2_smoke_v1 ]]; then
            [[ $# -eq 0 ]] || fail invalid_arguments \
                'rcq_v2_smoke_v1 accepts no caller-selected identity or runtime arguments'
            workspace="$(resolve_canonical_rcq_v2_workspace)"
            pin_parent="$workspace/qualification-pins"
            assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
            pin_parent="$DGX_SAFE_PATH"
            require_command flock
            open_private_lock "$pin_parent/.rcq-v2-reference-v2.lock" "$pin_parent" \
                'RCQ-v2 qualification authority lock'
            bind_rcq_v2_pretraining_authority "$workspace"
            release_id="$RCQ_PIN_RELEASE_ID"
            image="$RCQ_PIN_IMAGE_REFERENCE"
            config_path="$DGX_SMOKE_CONFIG"
            min_disk_gib=20
            min_memory_gib=16
            cpu_limit=2
            memory_limit_gib=8
        else
            [[ $# -eq 8 ]] || fail invalid_arguments \
                'smoke expects workspace release image config min_disk min_memory cpu_limit memory_limit'
            workspace="$(resolve_workspace "$1")"
            release_id="$2"
            image="$3"
            config_path="$4"
            min_disk_gib="$5"
            min_memory_gib="$6"
            cpu_limit="$7"
            memory_limit_gib="$8"
            assert_owned_workspace "$workspace"
            [[ "$config_path" != "$RCQ_REFERENCE_CONFIG" ]] ||
                fail dedicated_reference_required 'the RCQ-v2 reference configuration requires Invoke-DgxRcqV2Smoke.ps1'
        fi
        check_disk_gib "$workspace" "$min_disk_gib"
        check_memory_gib "$min_memory_gib" "$memory_limit_gib"
        check_gpu_idle
        check_container_image "$image"
        if [[ "$action" == rcq_v2_smoke_v1 && "$DGX_IMAGE_ID" != "$RCQ_PIN_IMAGE_ID" ]]; then
            fail image_id_mismatch 'cached container image differs from the pretraining pin'
        fi
        mkdir -p -- "$workspace/logs" "$workspace/runs" "$workspace/smoke-receipts" ||
            fail smoke_directory_failed 'could not create smoke artifact directories'
        container_smoke \
            "$workspace" "$release_id" "$image" "$config_path" \
            "$cpu_limit" "$memory_limit_gib" "$DGX_IMAGE_ID"
        ;;

    rcq_staging_canary)
        [[ $# -eq 4 ]] || fail invalid_arguments \
            'rcq_staging_canary expects min_disk min_memory cpu_limit memory_limit; all identities derive from the pretraining pin'
        min_disk_gib="$1"
        min_memory_gib="$2"
        cpu_limit="$3"
        memory_limit_gib="$4"
        workspace="$(resolve_canonical_rcq_v2_workspace)"
        pin_parent="$workspace/qualification-pins"
        assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
        pin_parent="$DGX_SAFE_PATH"
        require_command flock
        open_private_lock "$pin_parent/.rcq-v2-reference-v2.lock" "$pin_parent" \
            'RCQ-v2 qualification authority lock'
        bind_rcq_v2_pretraining_authority "$workspace"
        release_id="$RCQ_PIN_RELEASE_ID"
        image="$RCQ_PIN_IMAGE_REFERENCE"
        current_image_id="$RCQ_PIN_IMAGE_ID"
        pretraining_pin_file_sha="$RCQ_PIN_FILE_SHA256"
        pretraining_pin_sha="$RCQ_PIN_SEMANTIC_SHA256"
        run_id="rcq-staging-${pretraining_pin_file_sha:0:16}"
        release_root="$workspace/releases"
        assert_safe_directory_path "$release_root" "$workspace" 'release root'
        release_root="$DGX_SAFE_PATH"
        release="$release_root/$release_id"
        assert_safe_directory_path "$release" "$release_root" 'source release'
        release="$DGX_SAFE_PATH"
        assert_clean_release_python_source "$release"
        run_root="$workspace/runs"
        assert_safe_directory_path "$run_root" "$workspace" 'run root'
        run_root="$DGX_SAFE_PATH"
        logs_root="$workspace/logs"
        assert_safe_directory_path "$logs_root" "$workspace" 'log root'
        dataset_root="$workspace/datasets"
        assert_safe_directory_path "$dataset_root" "$workspace" 'dataset root'
        [[ -f "$release/$RCQ_STAGING_CANARY_CONFIG" && ! -L "$release/$RCQ_STAGING_CANARY_CONFIG" ]] ||
            fail missing_config "config '$RCQ_STAGING_CANARY_CONFIG' is missing or unsafe"
        [[ -f "$release/brain/src/irene_brain/training/train.py" && ! -L "$release/brain/src/irene_brain/training/train.py" ]] ||
            fail missing_trainer 'training module is missing or unsafe'
        require_slug release_id "$release_id"
        require_slug run_id "$run_id"
        [[ ! -e "$run_root/$run_id" && ! -L "$run_root/$run_id" ]] ||
            fail run_exists "run '$run_id' already exists; refusing overwrite"
        [[ ! -e "$logs_root/${run_id}.log" && ! -L "$logs_root/${run_id}.log" ]] ||
            fail log_exists "log for '$run_id' already exists; refusing overwrite"
        require_uint cpu_limit "$cpu_limit"
        require_uint container_memory_gib "$memory_limit_gib"
        (( cpu_limit >= 1 && memory_limit_gib >= 4 )) ||
            fail invalid_limit 'staging canary CPU and memory limits are too small'
        require_command flock
        require_command python3
        require_command timeout
        check_disk_gib "$workspace" "$min_disk_gib"
        check_memory_gib "$min_memory_gib" "$memory_limit_gib"
        check_container_image "$image"
        current_image_id="$DGX_IMAGE_ID"
        assert_run_inactive "$run_id"
        check_gpu_idle

        smoke_receipt_root="$workspace/smoke-receipts"
        assert_safe_directory_path "$smoke_receipt_root" "$workspace" 'smoke receipt root'
        smoke_receipt_root="$DGX_SAFE_PATH"
        smoke_receipt="$smoke_receipt_root/${release_id}.env"
        [[ -f "$smoke_receipt" && ! -L "$smoke_receipt" ]] ||
            fail missing_smoke_receipt 'this exact release has not passed the foreground smoke'
        grep -Fx "container_image=$image" "$smoke_receipt" >/dev/null ||
            fail smoke_image_mismatch 'smoke used a different container image'
        grep -Fx "container_image_id=$current_image_id" "$smoke_receipt" >/dev/null ||
            fail smoke_image_mismatch 'cached image content changed since smoke'
        grep -Fx 'workspace_contract=pseudo-brain-workspace-v2' "$smoke_receipt" >/dev/null ||
            fail smoke_workspace_mismatch 'smoke used a different workspace contract'

        canary_receipt_root="$workspace/rcq-staging-canary-receipts"
        mkdir -p -- "$canary_receipt_root" ||
            fail canary_receipt_directory_failed 'could not create the staging-canary receipt directory'
        assert_safe_directory_path "$canary_receipt_root" "$workspace" 'staging-canary receipt root'
        canary_receipt_root="$DGX_SAFE_PATH"
        canary_receipt="$canary_receipt_root/${release_id}.env"
        [[ ! -e "$canary_receipt" && ! -L "$canary_receipt" ]] ||
            fail staging_canary_already_passed 'this exact release already has a staging-canary receipt'

        container_uid="$(id -u)" || fail identity_probe_failed 'could not read container UID'
        container_gid="$(id -g)" || fail identity_probe_failed 'could not read container GID'
        require_uint container_uid "$container_uid"
        require_uint container_gid "$container_gid"
        write_rcq_staging_canary_job \
            "$workspace" "$release_id" "$image" "$run_id" \
            "$cpu_limit" "$memory_limit_gib" "$current_image_id" \
            "$container_uid" "$container_gid"
        job_path="$run_root/$run_id/launch.sh"
        run_metadata="$run_root/$run_id/run.env"
        config_sha="$(sha256sum "$release/$RCQ_STAGING_CANARY_CONFIG" | awk '{print $1}')" ||
            fail config_hash_failed 'could not hash the staging-canary configuration'
        launch_sha="$(sha256sum "$job_path" | awk '{print $1}')" ||
            fail launch_hash_failed 'could not hash the staging-canary launch script'
        created_utc="$(date -u +%Y%m%dT%H%M%SZ)" || fail clock_failed 'could not record run creation time'
        (
            set -o noclobber
            exec 8> "$run_metadata" || exit 75
            cat >&8 <<EOF
workspace_contract=pseudo-brain-workspace-v2
canary_contract=rcq-schema3-two-phase-v1
release_id=$release_id
container_image=$image
container_image_id=$current_image_id
pretraining_pin_file_sha256=$pretraining_pin_file_sha
pretraining_pin_sha256=$pretraining_pin_sha
config_path=$RCQ_STAGING_CANARY_CONFIG
config_sha256=$config_sha
created_utc=$created_utc
mode=foreground
min_free_disk_gib=$min_disk_gib
min_available_memory_gib=$min_memory_gib
container_cpu_count=$cpu_limit
container_memory_gib=$memory_limit_gib
container_uid=$container_uid
container_gid=$container_gid
network=none
ipc=host
pids_limit=512
cublas_workspace_config=:4096:8
launch_sha256=$launch_sha
EOF
        ) || fail run_metadata_exists 'staging-canary run metadata already exists; refusing overwrite'
        chmod 600 "$run_metadata" || fail metadata_permission_failed "could not protect '$run_metadata'"

        set +e
        timeout --signal=TERM --kill-after=15s 600s "$job_path"
        canary_status=$?
        set -e
        (( canary_status == 0 )) ||
            fail staging_canary_failed "foreground schema-3 staging canary exited with status $canary_status"

        run_dir="$run_root/$run_id"
        final_checkpoint="$run_dir/checkpoints/step-00000002.pt"
        metrics_path="$run_dir/metrics.jsonl"
        transition_path="$run_dir/stage-transition-01.json"
        invariance_path="$run_dir/invariance-step-00000002.json"
        for artifact in "$final_checkpoint" "$metrics_path" "$transition_path" "$invariance_path"; do
            assert_safe_regular_file "$artifact" "$run_dir" 'staging-canary evidence artifact'
        done
        final_checkpoint_sha="$(sha256sum "$final_checkpoint" | awk '{print $1}')"
        metrics_sha="$(sha256sum "$metrics_path" | awk '{print $1}')"
        transition_sha="$(sha256sum "$transition_path" | awk '{print $1}')"
        invariance_sha="$(sha256sum "$invariance_path" | awk '{print $1}')"
        completed_utc="$(date -u +%Y%m%dT%H%M%SZ)" || fail clock_failed 'could not record canary completion time'
        canary_receipt_tmp="${canary_receipt}.tmp.$$"
        trap 'rm -f -- "$canary_receipt_tmp"' EXIT
        (
            set -o noclobber
            exec 8> "$canary_receipt_tmp" || exit 75
            cat >&8 <<EOF
workspace_contract=pseudo-brain-workspace-v2
canary_contract=rcq-schema3-two-phase-v1
release_id=$release_id
container_image=$image
container_image_id=$current_image_id
pretraining_pin_file_sha256=$pretraining_pin_file_sha
pretraining_pin_sha256=$pretraining_pin_sha
config_path=$RCQ_STAGING_CANARY_CONFIG
config_sha256=$config_sha
run_id=$run_id
final_checkpoint=step-00000002.pt
final_checkpoint_sha256=$final_checkpoint_sha
metrics_sha256=$metrics_sha
stage_transition_sha256=$transition_sha
final_invariance_sha256=$invariance_sha
completed_utc=$completed_utc
EOF
        ) || fail canary_receipt_exists 'staging-canary receipt staging file already exists'
        chmod 600 "$canary_receipt_tmp" || fail metadata_permission_failed 'could not protect the staging-canary receipt'
        ln -- "$canary_receipt_tmp" "$canary_receipt" ||
            fail canary_receipt_exists 'staging-canary receipt appeared concurrently'
        rm -- "$canary_receipt_tmp"
        trap - EXIT
        printf 'rcq_staging_canary_receipt=%s\nrun_id=%s\n' "$canary_receipt" "$run_id"
        ;;

    train|rcq_v2_reference_train_v1)
        production_reference=false
        pretraining_pin_file_sha=none
        pretraining_pin_sha=none
        if [[ "$action" == rcq_v2_reference_train_v1 ]]; then
            [[ $# -eq 0 ]] || fail invalid_arguments \
                'rcq_v2_reference_train_v1 accepts no caller-selected identity or runtime arguments'
            production_reference=true
            workspace="$(resolve_canonical_rcq_v2_workspace)"
            pin_parent="$workspace/qualification-pins"
            assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
            pin_parent="$DGX_SAFE_PATH"
            require_command flock
            open_private_lock "$pin_parent/.rcq-v2-reference-v2.lock" "$pin_parent" \
                'RCQ-v2 qualification authority lock'
            bind_rcq_v2_pretraining_authority "$workspace"
            release_id="$RCQ_PIN_RELEASE_ID"
            image="$RCQ_PIN_IMAGE_REFERENCE"
            config_path="$RCQ_REFERENCE_CONFIG"
            run_id="$RCQ_REFERENCE_RUN_ID"
            mode=tmux
            min_disk_gib=20
            min_memory_gib=96
            cpu_limit=12
            memory_limit_gib=96
            pretraining_pin_file_sha="$RCQ_PIN_FILE_SHA256"
            pretraining_pin_sha="$RCQ_PIN_SEMANTIC_SHA256"
        else
            [[ $# -eq 10 ]] || fail invalid_arguments \
                'train expects workspace release image config run_id mode min_disk min_memory cpu_limit memory_limit'
            workspace="$(resolve_workspace "$1")"
            release_id="$2"
            image="$3"
            config_path="$4"
            run_id="$5"
            mode="$6"
            min_disk_gib="$7"
            min_memory_gib="$8"
            cpu_limit="$9"
            memory_limit_gib="${10}"
            assert_owned_workspace "$workspace"
            [[ "$config_path" != "$RCQ_REFERENCE_CONFIG" && "$run_id" != "$RCQ_REFERENCE_RUN_ID" ]] ||
                fail dedicated_reference_required 'the RCQ-v2 reference config/run requires Start-DgxRcqV2Reference.ps1'
        fi
        release_root="$workspace/releases"
        assert_safe_directory_path "$release_root" "$workspace" 'release root'
        release_root="$DGX_SAFE_PATH"
        release="$release_root/$release_id"
        assert_safe_directory_path "$release" "$release_root" 'source release'
        release="$DGX_SAFE_PATH"
        assert_clean_release_python_source "$release"
        run_root="$workspace/runs"
        assert_safe_directory_path "$run_root" "$workspace" 'run root'
        run_root="$DGX_SAFE_PATH"
        logs_root="$workspace/logs"
        assert_safe_directory_path "$logs_root" "$workspace" 'log root'
        dataset_root="$workspace/datasets"
        assert_safe_directory_path "$dataset_root" "$workspace" 'dataset root'
        require_relative_path config_path "$config_path"
        [[ "$config_path" != "$RCQ_STAGING_CANARY_CONFIG" ]] ||
            fail dedicated_canary_required 'the schema-3 staging canary must use Invoke-DgxRcqStagingCanary.ps1'
        [[ -f "$release/$config_path" && ! -L "$release/$config_path" ]] || fail missing_config "config '$config_path' is missing or unsafe"
        [[ -f "$release/brain/src/irene_brain/training/train.py" && ! -L "$release/brain/src/irene_brain/training/train.py" ]] || fail missing_trainer 'training module is missing or unsafe'
        require_slug run_id "$run_id"
        [[ "$mode" == foreground || "$mode" == tmux ]] || fail invalid_mode 'launch mode must be foreground or tmux'
        [[ ! -e "$run_root/$run_id" && ! -L "$run_root/$run_id" ]] || fail run_exists "run '$run_id' already exists; refusing overwrite"
        [[ ! -e "$logs_root/${run_id}.log" && ! -L "$logs_root/${run_id}.log" ]] || fail log_exists "log for '$run_id' already exists; refusing overwrite"
        require_command flock
        check_disk_gib "$workspace" "$min_disk_gib"
        check_memory_gib "$min_memory_gib" "$memory_limit_gib"
        check_container_image "$image"
        current_image_id="$DGX_IMAGE_ID"
        if [[ "$production_reference" == true && "$current_image_id" != "$RCQ_PIN_IMAGE_ID" ]]; then
            fail image_id_mismatch 'cached container image differs from the pretraining pin'
        fi
        assert_run_inactive "$run_id"
        check_gpu_idle
        receipt_root="$workspace/smoke-receipts"
        assert_safe_directory_path "$receipt_root" "$workspace" 'smoke receipt root'
        receipt_root="$DGX_SAFE_PATH"
        receipt="$receipt_root/${release_id}.env"
        [[ -f "$receipt" && ! -L "$receipt" ]] || fail missing_smoke_receipt 'this exact release has not passed the foreground smoke'
        grep -Fx "container_image=$image" "$receipt" >/dev/null || fail smoke_image_mismatch 'smoke used a different container image'
        grep -Fx "container_image_id=$current_image_id" "$receipt" >/dev/null || fail smoke_image_mismatch 'cached image content changed since smoke'
        grep -Fx 'workspace_contract=pseudo-brain-workspace-v2' "$receipt" >/dev/null || fail smoke_workspace_mismatch 'smoke used a different workspace contract'
        if [[ "$production_reference" == true ]]; then
            require_rcq_staging_canary_receipt \
                "$workspace" "$release" "$release_id" "$image" "$current_image_id" \
                "$pretraining_pin_file_sha" "$pretraining_pin_sha"
        fi
        require_uint cpu_limit "$cpu_limit"
        require_uint container_memory_gib "$memory_limit_gib"
        (( cpu_limit >= 1 && memory_limit_gib >= 4 )) || fail invalid_limit 'training CPU and memory limits are too small'
        container_uid="$(id -u)" || fail identity_probe_failed 'could not read container UID'
        container_gid="$(id -g)" || fail identity_probe_failed 'could not read container GID'
        require_uint container_uid "$container_uid"
        require_uint container_gid "$container_gid"
        write_training_job \
            "$workspace" "$release_id" "$image" "$config_path" "$run_id" \
            "$cpu_limit" "$memory_limit_gib" "$current_image_id" \
            "$container_uid" "$container_gid"
        job_path="$run_root/$run_id/launch.sh"
        run_metadata="$run_root/$run_id/run.env"
        config_sha="$(sha256sum "$release/$config_path" | awk '{print $1}')" ||
            fail config_hash_failed 'could not hash the training configuration'
        launch_sha="$(sha256sum "$job_path" | awk '{print $1}')" ||
            fail launch_hash_failed 'could not hash the training launch script'
        created_utc="$(date -u +%Y%m%dT%H%M%SZ)" || fail clock_failed 'could not record run creation time'
        (
            set -o noclobber
            exec 8> "$run_metadata" || exit 75
            cat >&8 <<EOF
workspace_contract=pseudo-brain-workspace-v2
release_id=$release_id
container_image=$image
container_image_id=$current_image_id
pretraining_pin_file_sha256=$pretraining_pin_file_sha
pretraining_pin_sha256=$pretraining_pin_sha
config_path=$config_path
config_sha256=$config_sha
created_utc=$created_utc
mode=$mode
min_free_disk_gib=$min_disk_gib
min_available_memory_gib=$min_memory_gib
container_cpu_count=$cpu_limit
container_memory_gib=$memory_limit_gib
container_uid=$container_uid
container_gid=$container_gid
network=none
ipc=host
pids_limit=2048
cublas_workspace_config=:4096:8
launch_sha256=$launch_sha
EOF
        ) || fail run_metadata_exists 'run metadata already exists; refusing overwrite'
        chmod 600 "$run_metadata" || fail metadata_permission_failed "could not protect '$run_metadata'"
        if [[ "$mode" == foreground ]]; then
            "$job_path" || fail training_failed "foreground training launch '$run_id' failed"
        else
            require_command tmux
            session="pseudo-brain-${run_id}"
            tmux has-session -t "$session" 2>/dev/null && fail session_exists "tmux session '$session' already exists"
            printf -v detached_command '%q ' \
                /usr/bin/env -i PATH=/usr/sbin:/usr/bin HOME=/nonexistent \
                /bin/bash -p -- "$job_path"
            tmux new-session -d \
                -e BASH_ENV= -e ENV= -e PYTHONHOME= -e PYTHONPATH= \
                -e PATH=/usr/sbin:/usr/bin -e HOME=/nonexistent \
                -s "$session" "$detached_command" ||
                fail tmux_start_failed "could not start tmux session '$session'"
            wait_for_detached_launch "$session" "$run_root/$run_id/launch.ready" "$run_id"
            printf 'tmux_session=%s\nrun_id=%s\n' "$session" "$run_id"
        fi
        ;;

    resume|rcq_v2_reference_resume_v1)
        production_reference=false
        pretraining_pin_file_sha=none
        pretraining_pin_sha=none
        if [[ "$action" == rcq_v2_reference_resume_v1 ]]; then
            [[ $# -eq 0 ]] || fail invalid_arguments \
                'rcq_v2_reference_resume_v1 accepts no caller-selected identity or runtime arguments'
            production_reference=true
            workspace="$(resolve_canonical_rcq_v2_workspace)"
            pin_parent="$workspace/qualification-pins"
            assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
            pin_parent="$DGX_SAFE_PATH"
            require_command flock
            open_private_lock "$pin_parent/.rcq-v2-reference-v2.lock" "$pin_parent" \
                'RCQ-v2 qualification authority lock'
            bind_rcq_v2_pretraining_authority "$workspace"
            release_id="$RCQ_PIN_RELEASE_ID"
            image="$RCQ_PIN_IMAGE_REFERENCE"
            config_path="$RCQ_REFERENCE_CONFIG"
            run_id="$RCQ_REFERENCE_RUN_ID"
            mode=tmux
            selection=latest
            checkpoint_name=auto
            checkpoint_sha=auto
            min_disk_gib=20
            min_memory_gib=96
            cpu_limit=12
            memory_limit_gib=96
            pretraining_pin_file_sha="$RCQ_PIN_FILE_SHA256"
            pretraining_pin_sha="$RCQ_PIN_SEMANTIC_SHA256"
        else
            [[ $# -eq 13 ]] || fail invalid_arguments \
                'resume expects workspace release image config run_id mode selection checkpoint sha min_disk min_memory cpu_limit memory_limit'
            workspace="$(resolve_workspace "$1")"
            release_id="$2"
            image="$3"
            config_path="$4"
            run_id="$5"
            mode="$6"
            selection="$7"
            checkpoint_name="$8"
            checkpoint_sha="$9"
            min_disk_gib="${10}"
            min_memory_gib="${11}"
            cpu_limit="${12}"
            memory_limit_gib="${13}"
            assert_owned_workspace "$workspace"
            [[ "$config_path" != "$RCQ_REFERENCE_CONFIG" && "$run_id" != "$RCQ_REFERENCE_RUN_ID" ]] ||
                fail dedicated_reference_required 'the RCQ-v2 reference config/run requires Resume-DgxRcqV2Reference.ps1'
        fi
        release_root="$workspace/releases"
        assert_safe_directory_path "$release_root" "$workspace" 'release root'
        release_root="$DGX_SAFE_PATH"
        release="$release_root/$release_id"
        assert_safe_directory_path "$release" "$release_root" 'source release'
        release="$DGX_SAFE_PATH"
        assert_clean_release_python_source "$release"
        require_relative_path config_path "$config_path"
        [[ "$config_path" != "$RCQ_STAGING_CANARY_CONFIG" ]] ||
            fail dedicated_canary_required 'the schema-3 staging canary must use Invoke-DgxRcqStagingCanary.ps1'
        [[ -f "$release/$config_path" && ! -L "$release/$config_path" ]] ||
            fail missing_config "config '$config_path' is missing or unsafe"
        [[ -f "$release/brain/src/irene_brain/training/train.py" && ! -L "$release/brain/src/irene_brain/training/train.py" ]] ||
            fail missing_trainer 'training module is missing or unsafe'
        require_slug run_id "$run_id"
        [[ "$mode" == foreground || "$mode" == tmux ]] || fail invalid_mode 'launch mode must be foreground or tmux'
        [[ "$selection" == latest || "$selection" == exact ]] ||
            fail invalid_checkpoint_selection 'selection must be latest or exact'
        require_uint cpu_limit "$cpu_limit"
        require_uint container_memory_gib "$memory_limit_gib"
        (( cpu_limit >= 1 && memory_limit_gib >= 4 )) || fail invalid_limit 'resume CPU and memory limits are too small'
        require_command flock
        require_command python3
        check_disk_gib "$workspace" "$min_disk_gib"
        check_memory_gib "$min_memory_gib" "$memory_limit_gib"
        check_container_image "$image"
        current_image_id="$DGX_IMAGE_ID"
        if [[ "$production_reference" == true && "$current_image_id" != "$RCQ_PIN_IMAGE_ID" ]]; then
            fail image_id_mismatch 'cached container image differs from the pretraining pin'
        fi
        container_uid="$(id -u)" || fail identity_probe_failed 'could not read container UID'
        container_gid="$(id -g)" || fail identity_probe_failed 'could not read container GID'
        require_uint container_uid "$container_uid"
        require_uint container_gid "$container_gid"
        verify_original_run_boundaries \
            "$workspace" "$release_id" "$image" "$config_path" "$run_id" \
            "$cpu_limit" "$memory_limit_gib" "$current_image_id" \
            "$min_disk_gib" "$min_memory_gib" "$container_uid" "$container_gid"
        if [[ "$production_reference" == true ]]; then
            run_metadata="$workspace/runs/$run_id/run.env"
            [[ "$(metadata_value "$run_metadata" pretraining_pin_file_sha256)" == "$pretraining_pin_file_sha" ]] ||
                fail pretraining_pin_mismatch 'reference run used a different pretraining pin file'
            [[ "$(metadata_value "$run_metadata" pretraining_pin_sha256)" == "$pretraining_pin_sha" ]] ||
                fail pretraining_pin_mismatch 'reference run used a different pretraining semantic pin'
        fi
        run_root="$workspace/runs"
        assert_safe_directory_path "$run_root" "$workspace" 'run root'
        run_root="$DGX_SAFE_PATH"
        run_dir="$run_root/$run_id"
        assert_safe_directory_path "$run_dir" "$run_root" 'owned run'
        run_dir="$DGX_SAFE_PATH"
        assert_run_inactive "$run_id"
        assert_run_lock_available "$run_dir"
        check_gpu_idle

        receipt_root="$workspace/smoke-receipts"
        assert_safe_directory_path "$receipt_root" "$workspace" 'smoke receipt root'
        receipt_root="$DGX_SAFE_PATH"
        receipt="$receipt_root/${release_id}.env"
        [[ -f "$receipt" && ! -L "$receipt" ]] ||
            fail missing_smoke_receipt 'the original release smoke receipt is missing or unsafe'
        grep -Fx "container_image=$image" "$receipt" >/dev/null ||
            fail smoke_image_mismatch 'smoke used a different container image'
        grep -Fx "container_image_id=$current_image_id" "$receipt" >/dev/null ||
            fail smoke_image_mismatch 'cached image content changed since smoke'
        grep -Fx 'workspace_contract=pseudo-brain-workspace-v2' "$receipt" >/dev/null ||
            fail smoke_workspace_mismatch 'smoke used a different workspace contract'
        if [[ "$production_reference" == true ]]; then
            require_rcq_staging_canary_receipt \
                "$workspace" "$release" "$release_id" "$image" "$current_image_id" \
                "$pretraining_pin_file_sha" "$pretraining_pin_sha"
        fi

        checkpoint_dir="$run_dir/checkpoints"
        assert_safe_directory_path "$checkpoint_dir" "$run_dir" 'owned checkpoint directory'
        checkpoint_dir="$DGX_SAFE_PATH"
        if [[ "$selection" == latest ]]; then
            [[ "$checkpoint_name" == auto && "$checkpoint_sha" == auto ]] ||
                fail invalid_checkpoint_selection 'latest selection cannot include caller checkpoint data'
            pointer_path="$checkpoint_dir/latest.json"
            [[ -f "$pointer_path" && ! -L "$pointer_path" ]] ||
                fail invalid_latest_pointer 'latest.json is missing or unsafe'
            set +e
            pointer_output="$(python3 -I - "$pointer_path" <<'PY'
import json
import re
import sys

def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    value = json.load(stream, object_pairs_hook=strict_object)
required = {"schema_version", "checkpoint", "checkpoint_sha256", "optimizer_step"}
if not isinstance(value, dict) or set(value) != required or value["schema_version"] != 1:
    raise ValueError("latest.json has incompatible fields")
name = value["checkpoint"]
digest = value["checkpoint_sha256"]
step = value["optimizer_step"]
if not isinstance(name, str) or re.fullmatch(r"step-[0-9]{8}\.pt", name) is None:
    raise ValueError("latest.json checkpoint name is invalid")
if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
    raise ValueError("latest.json checkpoint digest is invalid")
if type(step) is not int or step < 0 or name != f"step-{step:08d}.pt":
    raise ValueError("latest.json optimizer step does not match checkpoint")
print(f"{name}\t{digest}\t{step}")
PY
)"
            pointer_status=$?
            set -e
            (( pointer_status == 0 )) || fail invalid_latest_pointer 'latest.json failed strict validation'
            IFS=$'\t' read -r checkpoint_name checkpoint_sha pointer_step <<< "$pointer_output"
        else
            [[ "$checkpoint_name" =~ ^step-[0-9]{8}\.pt$ ]] ||
                fail invalid_checkpoint_name 'checkpoint must have the exact form step-00000000.pt'
            require_sha256 checkpoint_sha256 "$checkpoint_sha"
            pointer_step=''
        fi

        checkpoint_path="$checkpoint_dir/$checkpoint_name"
        [[ -f "$checkpoint_path" && ! -L "$checkpoint_path" ]] ||
            fail missing_checkpoint "checkpoint '$checkpoint_name' is missing or unsafe"
        resolved_checkpoint="$(realpath -e -- "$checkpoint_path")"
        resolved_checkpoint_dir="$(realpath -e -- "$checkpoint_dir")"
        [[ "$(dirname -- "$resolved_checkpoint")" == "$resolved_checkpoint_dir" ]] ||
            fail checkpoint_escape 'checkpoint escaped the owned run checkpoint directory'
        actual_checkpoint_sha="$(sha256sum "$resolved_checkpoint" | awk '{print $1}')"
        [[ "$actual_checkpoint_sha" == "$checkpoint_sha" ]] ||
            fail checkpoint_hash_mismatch 'checkpoint bytes do not match the required SHA-256'

        highest_checkpoint="$(
            find "$checkpoint_dir" -maxdepth 1 -type f -name 'step-????????.pt' -printf '%f\n' |
                grep -E '^step-[0-9]{8}\.pt$' |
                sort |
                tail -n 1
        )"
        [[ "$highest_checkpoint" == "$checkpoint_name" ]] ||
            fail checkpoint_not_highest 'resume refuses an older checkpoint because later checkpoint artifacts would be overwritten'
        step_digits="${checkpoint_name#step-}"
        step_digits="${step_digits%.pt}"
        checkpoint_step=$((10#$step_digits))
        if [[ -n "$pointer_step" ]]; then
            (( checkpoint_step == pointer_step )) ||
                fail invalid_latest_pointer 'latest.json step disagrees with checkpoint filename'
        fi

        set +e
        max_optimizer_steps="$(python3 -I - "$run_dir/resolved-config.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    value = json.load(stream)
maximum = value.get("run", {}).get("max_optimizer_steps")
if type(maximum) is not int or maximum < 1:
    raise ValueError("resolved config has no valid optimizer-step budget")
print(maximum)
PY
)"
        max_status=$?
        set -e
        (( max_status == 0 )) || fail invalid_run_metadata 'resolved configuration failed validation'
        require_uint max_optimizer_steps "$max_optimizer_steps"
        refuse_terminal_failed_stage_gate "$run_dir" "$checkpoint_step" "$release"
        (( checkpoint_step < max_optimizer_steps )) ||
            fail training_complete 'selected checkpoint already reached the configured optimizer-step budget'

        metrics_path="$run_dir/metrics.jsonl"
        if [[ -e "$metrics_path" || -L "$metrics_path" ]]; then
            assert_safe_regular_file "$metrics_path" "$run_dir" 'training metrics'
            metrics_path="$DGX_SAFE_PATH"
            metrics_bytes="$(wc -c < "$metrics_path")" || fail metrics_probe_failed 'could not read metrics byte boundary'
            metrics_lines="$(wc -l < "$metrics_path")" || fail metrics_probe_failed 'could not read metrics line boundary'
        else
            metrics_bytes=0
            metrics_lines=0
        fi
        require_uint metrics_bytes "$metrics_bytes"
        require_uint metrics_lines "$metrics_lines"

        mkdir -p -- "$run_dir/resume-attempts" || fail resume_directory_failed 'could not create resume-attempts directory'
        [[ -d "$run_dir/resume-attempts" && ! -L "$run_dir/resume-attempts" ]] ||
            fail unsafe_resume_directory 'resume-attempts is not an owned directory'
        attempt_id="resume-$(date -u +%Y%m%dT%H%M%SZ)-${checkpoint_sha:0:12}"
        require_slug attempt_id "$attempt_id"
        write_resume_job \
            "$workspace" "$release_id" "$image" "$config_path" "$run_id" \
            "$cpu_limit" "$memory_limit_gib" "$checkpoint_name" "$checkpoint_sha" \
            "$attempt_id" "$current_image_id" "$metrics_bytes" "$metrics_lines" \
            "$container_uid" "$container_gid"
        attempt_dir="$run_dir/resume-attempts/$attempt_id"
        job_path="$attempt_dir/launch.sh"
        resume_metadata="$attempt_dir/resume.env"
        config_sha="$(sha256sum "$release/$config_path" | awk '{print $1}')" ||
            fail config_hash_failed 'could not hash the training configuration'
        launch_sha="$(sha256sum "$job_path" | awk '{print $1}')" ||
            fail launch_hash_failed 'could not hash the resume launch script'
        created_utc="$(date -u +%Y%m%dT%H%M%SZ)" || fail clock_failed 'could not record resume creation time'
        (
            set -o noclobber
            exec 8> "$resume_metadata" || exit 75
            cat >&8 <<EOF
workspace_contract=pseudo-brain-workspace-v2
attempt_id=$attempt_id
release_id=$release_id
container_image=$image
container_image_id=$current_image_id
pretraining_pin_file_sha256=$pretraining_pin_file_sha
pretraining_pin_sha256=$pretraining_pin_sha
config_path=$config_path
config_sha256=$config_sha
run_id=$run_id
checkpoint=$checkpoint_name
checkpoint_sha256=$checkpoint_sha
checkpoint_step=$checkpoint_step
metrics_pre_resume_bytes=$metrics_bytes
metrics_pre_resume_lines=$metrics_lines
selection=$selection
mode=$mode
min_free_disk_gib=$min_disk_gib
min_available_memory_gib=$min_memory_gib
container_cpu_count=$cpu_limit
container_memory_gib=$memory_limit_gib
container_uid=$container_uid
container_gid=$container_gid
launch_sha256=$launch_sha
created_utc=$created_utc
EOF
        ) || fail resume_metadata_exists 'resume metadata already exists; refusing overwrite'
        chmod 600 "$resume_metadata" || fail metadata_permission_failed "could not protect '$resume_metadata'"
        if [[ "$mode" == foreground ]]; then
            "$job_path" || fail training_failed "foreground resume launch '$attempt_id' failed"
        else
            require_command tmux
            session="pseudo-brain-${run_id}"
            tmux has-session -t "$session" 2>/dev/null &&
                fail session_exists "tmux session '$session' already exists"
            printf -v detached_command '%q ' \
                /usr/bin/env -i PATH=/usr/sbin:/usr/bin HOME=/nonexistent \
                /bin/bash -p -- "$job_path"
            tmux new-session -d \
                -e BASH_ENV= -e ENV= -e PYTHONHOME= -e PYTHONPATH= \
                -e PATH=/usr/sbin:/usr/bin -e HOME=/nonexistent \
                -s "$session" "$detached_command" ||
                fail tmux_start_failed "could not start tmux session '$session'"
            wait_for_detached_launch "$session" "$attempt_dir/launch.ready" "$run_id"
            printf 'tmux_session=%s\n' "$session"
        fi
        printf 'resume_attempt=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\n' \
            "$attempt_id" "$checkpoint_name" "$checkpoint_sha"
        ;;

    status|historical_status)
        [[ $# -eq 2 ]] || fail invalid_arguments 'status expects workspace run_id'
        if [[ "$action" == historical_status ]]; then
            workspace="$(resolve_historical_workspace "$1")"
            assert_owned_historical_workspace "$workspace"
            runtime_prefix='irene-brain'
        else
            workspace="$(resolve_workspace "$1")"
            assert_owned_workspace "$workspace"
            runtime_prefix='pseudo-brain'
        fi
        run_id="$2"
        require_slug run_id "$run_id"
        run_root="$workspace/runs"
        run_dir="$run_root/$run_id"
        assert_safe_directory_path "$run_dir" "$run_root" 'run directory'
        run_dir="$DGX_SAFE_PATH"
        printf 'run_id=%s\n' "$run_id"
        if tmux has-session -t "${runtime_prefix}-${run_id}" 2>/dev/null; then
            printf 'tmux=running\n'
        else
            printf 'tmux=not-running\n'
        fi
        docker ps --filter "name=^/${runtime_prefix}-${run_id}$" --format 'container={{.Names}} status={{.Status}}'
        printf 'memory_available_gib=%s\n' "$(available_memory_gib)"
        printf 'disk_available_gib=%s\n' "$(report_available_disk_gib "$workspace")"
        nvidia-smi
        summarize_training_progress "$run_dir"
        log_path="$workspace/logs/${run_id}.log"
        if [[ -e "$log_path" || -L "$log_path" ]]; then
            assert_safe_regular_file "$log_path" "$workspace/logs" 'training log'
            log_path="$DGX_SAFE_PATH"
            printf '%s\n' '--- last 80 log lines ---'
            tail -n 80 -- "$log_path" || fail artifact_read_failed 'could not read training log'
        else
            printf 'log=not-created\n'
        fi
        resume_root="$run_dir/resume-attempts"
        if [[ -e "$resume_root" || -L "$resume_root" ]]; then
            assert_safe_directory_path "$resume_root" "$run_dir" 'resume-attempts directory'
            resume_root="$DGX_SAFE_PATH"
            latest_attempt="$(find "$resume_root" -mindepth 1 -maxdepth 1 -type d -name 'resume-*' -printf '%f\n' | sort | tail -n 1)"
            if [[ -n "$latest_attempt" ]]; then
                printf 'latest_resume_attempt=%s\n' "$latest_attempt"
                attempt_dir="$resume_root/$latest_attempt"
                assert_safe_directory_path "$attempt_dir" "$resume_root" 'resume attempt directory'
                resume_log="$DGX_SAFE_PATH/train.log"
                if [[ -e "$resume_log" || -L "$resume_log" ]]; then
                    assert_safe_regular_file "$resume_log" "$resume_root" 'resume log'
                    resume_log="$DGX_SAFE_PATH"
                    printf '%s\n' '--- last 80 resume log lines ---'
                    tail -n 80 -- "$resume_log" || fail artifact_read_failed 'could not read resume log'
                fi
            fi
        fi
        ;;

    artifact|historical_artifact)
        [[ $# -eq 3 ]] || fail invalid_arguments 'artifact expects workspace run_id logs|checkpoints|resume-attempts'
        if [[ "$action" == historical_artifact ]]; then
            workspace="$(resolve_historical_workspace "$1")"
            assert_owned_historical_workspace "$workspace"
        else
            workspace="$(resolve_workspace "$1")"
            assert_owned_workspace "$workspace"
        fi
        run_id="$2"
        kind="$3"
        require_slug run_id "$run_id"
        run_root="$workspace/runs"
        run_dir="$run_root/$run_id"
        assert_safe_directory_path "$run_dir" "$run_root" 'run directory'
        run_dir="$DGX_SAFE_PATH"
        case "$kind" in
            logs)
                path="$workspace/logs/${run_id}.log"
                assert_safe_regular_file "$path" "$workspace/logs" "log for '$run_id'"
                path="$DGX_SAFE_PATH"
                ;;
            checkpoints)
                path="$run_dir/checkpoints"
                assert_safe_artifact_tree "$path" "$run_dir" "checkpoint directory for '$run_id'"
                path="$DGX_SAFE_PATH"
                ;;
            resume-attempts)
                path="$run_dir/resume-attempts"
                assert_safe_artifact_tree "$path" "$run_dir" "resume attempts for '$run_id'"
                path="$DGX_SAFE_PATH"
                ;;
            *) fail invalid_artifact_kind 'artifact kind must be logs, checkpoints, or resume-attempts' ;;
        esac
        printf 'artifact_path=%s\n' "$path"
        ;;

    rcq_v2_authorize_final_v1)
        [[ $# -eq 5 ]] || fail invalid_arguments \
            'rcq_v2_authorize_final_v1 expects pretraining_pin_sha256 latest_pointer_sha256 entry_checkpoint_sha256 checkpoint_sha256 readiness_sha256; all paths derive from the frozen pins'
        workspace="$(resolve_canonical_rcq_v2_workspace)"
        pin_parent="$workspace/qualification-pins"
        assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
        pin_parent="$DGX_SAFE_PATH"
        require_command flock
        require_command python3
        open_private_lock "$pin_parent/.rcq-v2-reference-v2.lock" "$pin_parent" \
            'RCQ-v2 qualification authority lock'
        bind_rcq_v2_pretraining_authority "$workspace"
        open_rcq_v2_range_authority_lock "$workspace" "$RCQ_PIN_RANGE_CLAIM_ID"
        write_rcq_v2_final_authorization "$workspace" "$1" "$2" "$3" "$4" "$5"
        ;;

    rcq_v2_preclaim_v1)
        [[ $# -eq 0 ]] || fail invalid_arguments \
            'rcq_v2_preclaim_v1 accepts no caller-selected identity or runtime arguments'
        workspace="$(resolve_canonical_rcq_v2_workspace)"
        pin_parent="$workspace/qualification-pins"
        assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
        pin_parent="$DGX_SAFE_PATH"
        require_command flock
        open_private_lock "$pin_parent/.rcq-v2-reference-v2.lock" "$pin_parent" \
            'RCQ-v2 qualification authority lock'
        bind_rcq_v2_pretraining_authority "$workspace"
        open_rcq_v2_range_authority_lock "$workspace" "$RCQ_PIN_RANGE_CLAIM_ID"
        prepare_rcq_v2_evaluator_paths "$workspace" preclaim
        run_rcq_v2_evaluator_container preclaim
        ;;

    rcq_v2_final_once_v1)
        [[ $# -eq 0 ]] || fail invalid_arguments \
            'rcq_v2_final_once_v1 accepts no caller-selected identity or runtime arguments'
        workspace="$(resolve_canonical_rcq_v2_workspace)"
        pin_parent="$workspace/qualification-pins"
        assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
        pin_parent="$DGX_SAFE_PATH"
        require_command flock
        open_private_lock "$pin_parent/.rcq-v2-reference-v2.lock" "$pin_parent" \
            'RCQ-v2 qualification authority lock'
        bind_rcq_v2_pretraining_authority "$workspace"
        open_rcq_v2_range_authority_lock "$workspace" "$RCQ_PIN_RANGE_CLAIM_ID"
        prepare_rcq_v2_evaluator_paths "$workspace" final-once
        run_rcq_v2_evaluator_container final-once
        ;;

    rcq_v2_verify_receipt_v1)
        [[ $# -eq 0 ]] || fail invalid_arguments \
            'rcq_v2_verify_receipt_v1 accepts no caller-selected identity or runtime arguments'
        workspace="$(resolve_canonical_rcq_v2_workspace)"
        pin_parent="$workspace/qualification-pins"
        assert_account_owned_directory "$pin_parent" "$workspace" 'qualification pin root' 700
        pin_parent="$DGX_SAFE_PATH"
        require_command flock
        open_private_lock "$pin_parent/.rcq-v2-reference-v2.lock" "$pin_parent" \
            'RCQ-v2 qualification authority lock'
        bind_rcq_v2_receipt_verification_authority "$workspace"
        open_rcq_v2_range_authority_lock "$workspace" "$RCQ_PIN_RANGE_CLAIM_ID"
        prepare_rcq_v2_evaluator_paths "$workspace" verify-receipt
        run_rcq_v2_evaluator_container verify-receipt
        ;;

    *) fail unknown_action "unsupported action '$action'" ;;
esac
