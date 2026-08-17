"""Derive the RCQ-v3 evaluator lineage from the frozen RCQ-v2 evaluator files.

The historical RCQ-v2 evaluator bundle (``evaluation/rcq_v2.py``,
``evaluation/rcq_v2_final.py``, ``evaluation/rcq_v2_torch.py``) is pinned by
digest inside the immutable v2 registrations and must never be edited.  The v3
lineage is therefore a new evaluator bundle:

- ``evaluation/rcq_v2.py`` stays shared (identical gate definitions and
  thresholds; the D5 decision keeps the frozen gate IDs and every threshold).
- ``evaluation/rcq_v3_final.py`` and ``evaluation/rcq_v3_torch.py`` are
  create-only copies of the v2 modules with exactly the identity substitutions
  tabulated below.
- ``evaluation/rcq_v3_registration.py`` is the matching target-blind
  preregistration builder.

Every substitution is an exact byte replacement with an asserted occurrence
count, so this script fails loudly instead of silently drifting from the frozen
v2 text.  After writing, it prints a residual census of v2-identity tokens so a
reviewer can confirm that every remaining mention is deliberate (shared gate
IDs, shared check-module imports, the stable range-retirement protocol string,
and unchanged function names).

Frozen v3 identity decisions (see brain/docs/ROADMAP_TO_PACMAN.md and the v3
preregistration run record):

- qualification id ``rcq_v3_reference_v1``; evaluator id ``rcq_v3_final_v1``;
  run id ``dgx-rcq-v3-reference-seed-1702`` (D4: seed 1702 reused; only the
  objective recipe changes relative to the v2 reference).
- Fresh sealed TEST namespace, local to the ``test`` split:
  recipient ``[4194304, 4194816)``, donor ``[4194816, 4195328)``, guard
  ``[4195328, 4195840)`` -- the next unused 2^20-aligned family after the v2
  ``[3145728, 3147264)`` family.  The v1 retired ranges, the v2 ranges, and the
  matched-campaign reservation ``[2097152, 2097920)`` stay untouched forever.
- The range-retirement protocol string
  ``irene_moving_shapes_test_range_retirement_v1`` is deliberately unchanged:
  any future evaluator reusing sealed ranges must collide on the claim key.
- Gate IDs ``rcq_v2_development_v1`` / ``rcq_v2_value_development_v1`` and every
  registered threshold are unchanged (D5); the recipe change lives in the v3
  configuration hash, not in the gates.
- Dispatcher action names move to ``rcq_v3_*`` so the trusted dispatcher can
  hold v2 (closed) and v3 authority side by side.

Usage: ``python brain/scripts/build_rcq_v3_lineage.py [--force]`` from anywhere.
Targets are create-only unless ``--force`` is given.

``--dispatcher`` derives the RCQ-v3 qualification family inside the trusted
host dispatcher ``brain/scripts/dgx/_remote_dispatch.sh``: nine v3 sibling
functions and five mechanical v3 action cases are extracted from their frozen
v2 counterparts and rewritten with the asserted substitutions below, then
inserted between the section markers.  The shared structural edits (case-label
extensions, canary-config parameterization, the release-sync whitelist, and
the v3 constants block) are hand-reviewed in version control; this mode
refuses to run unless those anchors are already present, and it refuses to
run twice.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION = REPO_ROOT / "brain" / "src" / "irene_brain" / "evaluation"
DISPATCHER = REPO_ROOT / "brain" / "scripts" / "dgx" / "_remote_dispatch.sh"

# (old, new, expected count) — count None means "report only, at least one".
FINAL_SUBSTITUTIONS = [
    ('FINAL_EVALUATOR_ID = "rcq_v2_final_v1"',
     'FINAL_EVALUATOR_ID = "rcq_v3_final_v1"', 1),
    ("RECIPIENT_OFFSET = 3_145_728", "RECIPIENT_OFFSET = 4_194_304", 1),
    ("DONOR_OFFSET = 3_146_240", "DONOR_OFFSET = 4_194_816", 1),
    ("RETIRED_END = 3_146_752", "RETIRED_END = 4_195_328", 1),
    ("GUARD_END = 3_147_264", "GUARD_END = 4_195_840", 1),
    ('"evaluation/rcq_v2_final.py",', '"evaluation/rcq_v3_final.py",', 1),
    ('"evaluation/rcq_v2_torch.py",', '"evaluation/rcq_v3_torch.py",', 1),
    ("irene_brain.evaluation.rcq_v2_torch",
     "irene_brain.evaluation.rcq_v3_torch", 1),
    ('"rcq_v2_reference_v2"', '"rcq_v3_reference_v1"', 1),
    ('"dgx-rcq-v2-reference-seed-1702"',
     '"dgx-rcq-v3-reference-seed-1702"', 1),
    ('"rcq_v2_final_once_v1"', '"rcq_v3_final_once_v1"', 1),
    ('"rcq_v2_preclaim_v1"', '"rcq_v3_preclaim_v1"', 1),
    ('"rcq_v2_verify_receipt_v1"', '"rcq_v3_verify_receipt_v1"', 1),
    ("rcq-v2-reference-v2", "rcq-v3-reference-v1", 3),
    ('"reserved_first_matched_evaluation_suite_excluded_from_rcq_v2"',
     '"reserved_first_matched_evaluation_suite_excluded_from_rcq_v3"', 3),
    ('"id": "rcq_v2_final_recipient"', '"id": "rcq_v3_final_recipient"', 1),
    ('"id": "rcq_v2_final_donor"', '"id": "rcq_v3_final_donor"', 1),
    ('"id": "rcq_v2_final_guard"', '"id": "rcq_v3_final_guard"', 1),
    # The v3 namespace ledger must also record the sealed v2 family: v2 failed
    # terminally at its development entry gate before any claim, so its ranges
    # are retired unclaimed and stay excluded forever.  This insertion runs
    # after the id renames above and anchors on the renamed v3 recipient entry.
    (
        '        {\n            "id": "rcq_v3_final_recipient",',
        '        {\n'
        '            "id": "rcq_v2_final_recipient",\n'
        '            "split": "test",\n'
        '            "local_start": 3_145_728,\n'
        '            "local_end": 3_146_240,\n'
        '            "status": "sealed_unclaimed_retired_after_terminal_entry_failure",\n'
        '        },\n'
        '        {\n'
        '            "id": "rcq_v2_final_donor",\n'
        '            "split": "test",\n'
        '            "local_start": 3_146_240,\n'
        '            "local_end": 3_146_752,\n'
        '            "status": "sealed_unclaimed_retired_after_terminal_entry_failure",\n'
        '        },\n'
        '        {\n'
        '            "id": "rcq_v2_final_guard",\n'
        '            "split": "test",\n'
        '            "local_start": 3_146_752,\n'
        '            "local_end": 3_147_264,\n'
        '            "status": "sealed_unused_excluded",\n'
        '        },\n'
        '        {\n'
        '            "id": "rcq_v3_final_recipient",',
        1,
    ),
    ("RCQ-v2", "RCQ-v3", None),
]

TORCH_SUBSTITUTIONS = [
    ("from .rcq_v2_final import (", "from .rcq_v3_final import (", 1),
    ("rcq-v2-final.", "rcq-v3-final.", 5),
    ('"brain/configs/training/dgx-rcq-v2-reference.toml"',
     '"brain/configs/training/dgx-rcq-v3-reference.toml"', 1),
    ("rcq-v2-reference-v2", "rcq-v3-reference-v1", 5),
    ('"rcq_v2_pin_pretraining_v1"', '"rcq_v3_pin_pretraining_v1"', 1),
    ('"rcq_v2_reference_v2"', '"rcq_v3_reference_v1"', 3),
    ("dgx-rcq-v2-reference-seed-1702",
     "dgx-rcq-v3-reference-seed-1702", 3),
    ('"rcq_v2_authorize_final_v1"', '"rcq_v3_authorize_final_v1"', 1),
    ('"rcq_v2_preclaim_v1"', '"rcq_v3_preclaim_v1"', 3),
    ('"rcq_v2_final_v1"', '"rcq_v3_final_v1"', 1),
    ('"rcq_v2_verify_receipt_v1"', '"rcq_v3_verify_receipt_v1"', 4),
    ("_claimed_rcq_v2_test_dataset", "_claimed_rcq_v3_test_dataset", 2),
    ("RCQ-v2", "RCQ-v3", None),
]

REGISTRATION_SUBSTITUTIONS = [
    ("from .rcq_v2_final import (", "from .rcq_v3_final import (", 1),
    ("from .rcq_v2_torch import (", "from .rcq_v3_torch import (", 1),
    ('"rcq_v2_reference_v2"', '"rcq_v3_reference_v1"', 1),
    ('"rcq_v2_final_once_v1"', '"rcq_v3_final_once_v1"', 1),
    ('"rcq_v2_preclaim_v1"', '"rcq_v3_preclaim_v1"', 1),
    ('"rcq_v2_verify_receipt_v1"', '"rcq_v3_verify_receipt_v1"', 1),
    ("rcq-v2-reference-v2", "rcq-v3-reference-v1", 3),
    ("RCQ-v2", "RCQ-v3", None),
]

TARGETS = [
    ("rcq_v2_final.py", "rcq_v3_final.py", FINAL_SUBSTITUTIONS),
    ("rcq_v2_torch.py", "rcq_v3_torch.py", TORCH_SUBSTITUTIONS),
    ("rcq_v2_registration.py", "rcq_v3_registration.py", REGISTRATION_SUBSTITUTIONS),
]

RESIDUAL_TOKENS = (
    "rcq_v2",
    "rcq-v2",
    "RCQ-v2",
    "3_145_728",
    "3_146_240",
    "3_146_752",
    "3_147_264",
    "3145728",
    "3146240",
    "3146752",
    "3147264",
    "dgx-rcq-v2",
)

# --------------------------------------------------------------------------
# Trusted host dispatcher derivation (--dispatcher)
# --------------------------------------------------------------------------

DISPATCHER_FUNCTIONS = [
    "assert_canonical_rcq_v2_registration_file",
    "load_rcq_v2_pretraining_pin_summary",
    "verify_rcq_v2_pretraining_host_bindings",
    "bind_rcq_v2_pretraining_authority",
    "write_rcq_v2_pretraining_pin",
    "write_rcq_v2_final_authorization",
    "bind_rcq_v2_receipt_verification_authority",
    "prepare_rcq_v2_evaluator_paths",
    "run_rcq_v2_evaluator_container",
]

DISPATCHER_MECHANICAL_CASES = [
    "rcq_v2_pin_pretraining_v1",
    "rcq_v2_authorize_final_v1",
    "rcq_v2_preclaim_v1",
    "rcq_v2_final_once_v1",
    "rcq_v2_verify_receipt_v1",
]

# Ordered exact replacements for dispatcher text.  The two qualification-
# neutral helpers keep their historical names everywhere (the canonical
# workspace resolver and the claim-id-parameterized range authority lock), so
# they are protected with placeholders before the blanket renames and restored
# at the end.
DISPATCHER_SUBSTITUTIONS = [
    ("resolve_canonical_rcq_v2_workspace", "RCQV2WORKSPACERESOLVER"),
    ("open_rcq_v2_range_authority_lock", "RCQV2RANGEAUTHORITYLOCK"),
    ("_rcq_v2_", "_rcq_v3_"),
    ("rcq_v2_reference_v2", "rcq_v3_reference_v1"),
    ("rcq-v2-reference-v2", "rcq-v3-reference-v1"),
    ("dgx-rcq-v2-reference-seed-1702", "dgx-rcq-v3-reference-seed-1702"),
    ("dgx-rcq-v2-reference.toml", "dgx-rcq-v3-reference.toml"),
    ("rcq_v2_pin_pretraining_v1", "rcq_v3_pin_pretraining_v1"),
    ("rcq_v2_preclaim_v1", "rcq_v3_preclaim_v1"),
    ("rcq_v2_authorize_final_v1", "rcq_v3_authorize_final_v1"),
    ("rcq_v2_final_once_v1", "rcq_v3_final_once_v1"),
    ("rcq_v2_verify_receipt_v1", "rcq_v3_verify_receipt_v1"),
    ("pseudo-brain-rcq-v2-", "pseudo-brain-rcq-v3-"),
    ("RCQ_PRECLAIM_ISOLATED_BOOTSTRAP", "RCQ_V3_PRECLAIM_ISOLATED_BOOTSTRAP"),
    ("RCQ_FINAL_ONCE_ISOLATED_BOOTSTRAP", "RCQ_V3_FINAL_ONCE_ISOLATED_BOOTSTRAP"),
    (
        "RCQ_VERIFY_RECEIPT_ISOLATED_BOOTSTRAP",
        "RCQ_V3_VERIFY_RECEIPT_ISOLATED_BOOTSTRAP",
    ),
    ("RCQ_REGISTRATION_RELATIVE_PATH", "RCQ_V3_REGISTRATION_RELATIVE_PATH"),
    ("RCQ_READINESS_RELATIVE_PATH", "RCQ_V3_READINESS_RELATIVE_PATH"),
    ("RCQ_REFERENCE_CONFIG", "RCQ_V3_REFERENCE_CONFIG"),
    ("RCQ_REFERENCE_RUN_ID", "RCQ_V3_REFERENCE_RUN_ID"),
    ("RCQ-v2", "RCQ-v3"),
    ("RCQV2WORKSPACERESOLVER", "resolve_canonical_rcq_v2_workspace"),
    ("RCQV2RANGEAUTHORITYLOCK", "open_rcq_v2_range_authority_lock"),
]

# Hand-reviewed structural edits that must already be present in the
# dispatcher before the derived family is inserted.
DISPATCHER_REQUIRED_ANCHORS = [
    "RCQ_V3_REFERENCE_CONFIG='brain/configs/training/dgx-rcq-v3-reference.toml'",
    "RCQ_V3_REFERENCE_RUN_ID='dgx-rcq-v3-reference-seed-1702'",
    "smoke|rcq_v2_smoke_v1|rcq_v3_smoke_v1)",
    "rcq_staging_canary|rcq_v3_staging_canary)",
    "train|rcq_v2_reference_train_v1|rcq_v3_reference_train_v1)",
    "resume|rcq_v2_reference_resume_v1|rcq_v3_reference_resume_v1)",
    "assert_canonical_rcq_v3_registration_file",
]

# Anchors that legitimately appear more than once (one per v3 action case
# or per release-sync whitelist site).
DISPATCHER_REQUIRED_MULTI_ANCHORS = [
    "bind_rcq_v3_pretraining_authority",
    "registrations/rcq-v3-reference-v1.json",
]

DISPATCHER_SECTION_BEGIN = (
    "# --- begin derived RCQ-v3 qualification family "
    "(brain/scripts/build_rcq_v3_lineage.py --dispatcher) ---"
)
DISPATCHER_SECTION_END = (
    "# --- end derived RCQ-v3 qualification family ---"
)


def _extract_function(text: str, name: str) -> str:
    start_token = f"\n{name}() {{\n"
    if text.count(start_token) != 1:
        raise SystemExit(
            f"dispatcher: expected exactly one definition of {name}(); "
            "the frozen dispatcher text drifted"
        )
    start = text.index(start_token) + 1
    # The body may embed Python heredocs whose lines include a bare ``}``, so a
    # plain ``\n}\n`` search truncates early.  The real end of a dispatcher
    # function is a ``}`` line directly followed by the next function
    # definition or by the action-table anchor.
    end_pattern = re.compile(
        r"\n\}\n(?=\n?(?:[a-z0-9_]+\(\) \{|action=\"\$\{1:-\}\"))"
    )
    match = end_pattern.search(text, start)
    if match is None:
        raise SystemExit(f"dispatcher: cannot find the end of {name}()")
    return text[start : match.start() + 3]


def _extract_case(text: str, label: str) -> str:
    start_token = f"    {label})\n"
    if text.count(start_token) != 1:
        raise SystemExit(
            f"dispatcher: expected exactly one action case {label}); "
            "the frozen dispatcher text drifted"
        )
    start = text.index(start_token)
    end = text.find("\n        ;;\n", start)
    if end == -1:
        raise SystemExit(f"dispatcher: cannot find the end of case {label})")
    return text[start : end + len("\n        ;;\n")]


def _substitute_dispatcher(text: object) -> str:
    result = str(text)
    for old, new in DISPATCHER_SUBSTITUTIONS:
        result = result.replace(old, new)
    for forbidden in (
        "rcq_v2_reference_v2",
        "rcq-v2-reference-v2",
        "dgx-rcq-v2",
        "rcq_v2_pin_pretraining_v1",
        "rcq_v2_preclaim_v1",
        "rcq_v2_authorize_final_v1",
        "rcq_v2_final_once_v1",
        "rcq_v2_verify_receipt_v1",
        "RCQ-v2",
        "RCQ_REFERENCE_CONFIG",
        "RCQ_REFERENCE_RUN_ID",
        "RCQ_REGISTRATION_RELATIVE_PATH",
        "RCQ_READINESS_RELATIVE_PATH",
    ):
        if forbidden in result:
            raise SystemExit(
                f"dispatcher derivation left a forbidden v2 token {forbidden!r}"
            )
    return result


def _derive_dispatcher() -> None:
    text = DISPATCHER.read_text(encoding="utf-8")
    if DISPATCHER_SECTION_BEGIN in text:
        raise SystemExit(
            "dispatcher already contains the derived RCQ-v3 family; refusing a second insert"
        )
    for anchor in DISPATCHER_REQUIRED_ANCHORS:
        if text.count(anchor) != 1:
            raise SystemExit(
                f"dispatcher: expected exactly one hand-reviewed anchor {anchor!r}; "
                "apply the structural edits first"
            )
    for anchor in DISPATCHER_REQUIRED_MULTI_ANCHORS:
        if anchor not in text:
            raise SystemExit(
                f"dispatcher: missing hand-reviewed anchor {anchor!r}; "
                "apply the structural edits first"
            )
    functions = [_extract_function(text, name) for name in DISPATCHER_FUNCTIONS]
    cases = [_extract_case(text, label) for label in DISPATCHER_MECHANICAL_CASES]
    derived_functions = [_substitute_dispatcher(block) for block in functions]
    derived_cases = [_substitute_dispatcher(block) for block in cases]

    function_anchor = '\naction="${1:-}"\n'
    if text.count(function_anchor) != 1:
        raise SystemExit("dispatcher: the action-table anchor is not unique")
    case_anchor = '    *) fail unknown_action "unsupported action \'$action\'" ;;\n'
    if text.count(case_anchor) != 1:
        raise SystemExit("dispatcher: the unknown-action anchor is not unique")

    function_section = (
        "\n"
        + DISPATCHER_SECTION_BEGIN
        + "\n"
        + "\n".join(block.rstrip("\n") for block in derived_functions)
        + "\n"
        + DISPATCHER_SECTION_END
        + "\n"
    )
    case_section = (
        "\n"
        + DISPATCHER_SECTION_BEGIN
        + "\n\n"
        + "\n".join(block.rstrip("\n") for block in derived_cases)
        + "\n"
        + DISPATCHER_SECTION_END
        + "\n"
    )
    text = text.replace(function_anchor, function_section + function_anchor, 1)
    text = text.replace(case_anchor, case_section + case_anchor, 1)
    DISPATCHER.write_text(text, encoding="utf-8", newline="")
    print(
        "derived the RCQ-v3 dispatcher family: "
        f"{len(derived_functions)} functions, {len(derived_cases)} action cases"
    )


def _apply(source_name: str, target_name: str, substitutions: list) -> None:
    source_path = EVALUATION / source_name
    target_path = EVALUATION / target_name
    text = source_path.read_text(encoding="utf-8")
    for old, new, expected in substitutions:
        found = text.count(old)
        if expected is not None and found != expected:
            raise SystemExit(
                f"{source_name}: expected {expected} occurrence(s) of {old!r}, "
                f"found {found}; the frozen v2 text drifted -- do not proceed"
            )
        if expected is None and found == 0:
            raise SystemExit(
                f"{source_name}: expected at least one occurrence of {old!r}"
            )
        text = text.replace(old, new)
    if target_path.exists():
        if not "--force" in sys.argv[1:]:
            raise SystemExit(
                f"{target_path} already exists; pass --force to re-derive"
            )
        target_path.unlink()
    target_path.write_text(text, encoding="utf-8", newline="")
    print(f"derived {target_path.relative_to(REPO_ROOT)} from {source_name}")


def _residual_census(target_name: str) -> None:
    lines = (EVALUATION / target_name).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if any(token in line for token in RESIDUAL_TOKENS):
            print(f"  residual {target_name}:{number}: {line.strip()}")


def main() -> int:
    if "--dispatcher" in sys.argv[1:]:
        _derive_dispatcher()
        return 0
    for source_name, target_name, substitutions in TARGETS:
        _apply(source_name, target_name, substitutions)
    print("\nresidual v2-identity census (every line must be deliberate):")
    for _source_name, target_name, _substitutions in TARGETS:
        _residual_census(target_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
