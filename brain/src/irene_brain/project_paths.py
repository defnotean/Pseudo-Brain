"""CWD-independent path resolution for Pseudo-Brain command-line tools."""

from __future__ import annotations

import os
import stat
from pathlib import Path


BRAIN_ROOT = Path(__file__).resolve().parents[2]


def _is_reparse(path: Path) -> bool:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _reject_reparse_components(path: Path) -> None:
    components = list(path.parents)
    components.reverse()
    components.append(path)
    for component in components:
        if os.path.lexists(component) and _is_reparse(component):
            raise ValueError(
                f"workspace path cannot traverse a symlink or reparse point: {component}"
            )


def resolve_workspace_path(
    path: str | os.PathLike[str],
    *,
    brain_root: Path = BRAIN_ROOT,
) -> Path:
    """Resolve relative paths below ``brain_root`` without consulting the CWD.

    Absolute paths remain valid for explicit external locations such as DGX
    bind mounts. Both forms reject parent traversal and existing symlink or
    reparse-point components.
    """

    candidate = Path(path)
    if not candidate.parts:
        raise ValueError("workspace path must not be empty")
    if any(part == ".." for part in candidate.parts):
        raise ValueError("workspace path cannot contain parent traversal")

    if candidate.is_absolute():
        resolved = Path(os.path.abspath(candidate))
    else:
        if candidate.anchor or candidate.drive or any(
            ":" in part for part in candidate.parts
        ):
            raise ValueError("relative workspace path has an invalid anchor")
        root = Path(os.path.abspath(brain_root))
        resolved = Path(os.path.abspath(root.joinpath(*candidate.parts)))
        try:
            resolved.relative_to(root)
        except ValueError as error:  # pragma: no cover - guarded above, kept fail-closed
            raise ValueError("relative workspace path escapes the brain root") from error

    _reject_reparse_components(resolved)
    return resolved


__all__ = ["BRAIN_ROOT", "resolve_workspace_path"]
