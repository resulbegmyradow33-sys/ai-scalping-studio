"""No provider key, private key or credentialed URL is committed.

The scan reads only files git tracks, so a developer's untracked ``.env`` or a
local key never fails the suite. Patterns are deliberately high-signal — a
provider prefix or a private-key header, not generic entropy — so a hit is
treated as real and the gate keeps its credibility. A finding reports the file,
line and rule, never the value itself.

The audit that motivated this test is ``docs/secret-audit-2026-11-24.md``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

#: ``backend/tests/test_no_committed_secrets.py`` -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Files that name secret formats on purpose rather than containing one.
ALLOWED = frozenset(
    {
        "backend/tests/test_no_committed_secrets.py",
        "docs/secret-audit-2026-11-24.md",
    }
)

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".next",
        "node_modules",
        "out",
        "coverage",
        "test-results",
        "playwright-report",
        ".worktrees",
        "runs",
    }
)

SKIP_NAMES = frozenset(
    {
        "package-lock.json",
        "uv.lock",
        "pnpm-lock.yaml",
        "yarn.lock",
        ".DS_Store",
    }
)

SKIP_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".woff",
        ".woff2",
        ".ttf",
        ".parquet",
        ".pdf",
        ".pyc",
        ".lock",
    }
)

#: High-signal formats only. Generic entropy stays in gitleaks, not here.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("stripe-live-key", re.compile(r"\bsk_live_[A-Za-z0-9]{10,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    ),
    (
        "credentials-in-url",
        re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s@]+@[^\s/\"'`]+"),
    ),
)

#: Grow the gate without turning it into a wall: a bounded read keeps it fast.
MAX_BYTES = 2_000_000


def _tracked_files() -> list[str]:
    """Return repository-relative paths of every file git tracks."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - env dependent
        pytest.skip("git is unavailable; the tracked-file scan cannot run")
    return [name for name in completed.stdout.decode("utf-8", "replace").split("\0") if name]


def _is_scannable(relative: str, path: Path) -> bool:
    if relative in ALLOWED:
        return False
    if set(Path(relative).parts) & SKIP_DIRS:
        return False
    if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
        return False
    return path.is_file()


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def test_no_committed_secret_material() -> None:
    tracked = _tracked_files()
    # A broken walk must fail loudly instead of reporting a clean scan.
    assert len(tracked) > 100, (
        f"only {len(tracked)} tracked files found; the scan is not meaningful"
    )

    findings: list[str] = []
    for relative in tracked:
        path = REPO_ROOT / relative
        if not _is_scannable(relative, path):
            continue
        text = _read_text(path)
        if text is None:
            continue
        for rule, pattern in PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line} [{rule}]")

    assert not findings, "committed secret-like values (value redacted):\n" + "\n".join(findings)
