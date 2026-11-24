"""The repository's ignore rules cover the secret surface, without hiding tracked files.

F-2 in ``docs/secret-audit-2026-11-24.md``: ``.gitignore`` ignored only ``.env``,
so ``.env.local``, private keys, ``.npmrc`` and credential JSON/YAML files were
all reachable by a plain ``git add -A``. The widened rules are pinned here so a
later edit cannot quietly reopen that gap, and so a rule can never make a
tracked file vanish from ``git status``.

Every check runs ``git check-ignore --no-index``, which reports what the rules
say about a path regardless of whether that path exists or is in the index. The
paths below are representative, deliberately not created on disk.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

#: ``backend/tests/test_gitignore_hygiene.py`` -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Paths a developer, editor or tool realistically drops into the tree.
#: Matched against the ignore rules only; none of these files exists today.
SECRET_PATHS: tuple[str, ...] = (
    ".env",
    ".env.local",
    ".env.production",
    "backend/.env.staging",
    "id_rsa",
    "id_ed25519",
    "deploy/id_rsa",
    "certs/server.pem",
    "certs/server.key",
    "secrets.json",
    "credentials.yml",
    "credentials.yaml",
    ".npmrc",
    ".netrc",
    "wallet.p12",
    "wallet.pfx",
    "release.keystore",
    "release.jks",
)

#: The one safe template, tracked on purpose so it must stay visible.
EXAMPLE_PATH = "frontend/.env.example"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git at the repository root; skip when git is unavailable."""
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - environment dependent
        pytest.skip("git is unavailable; ignore rules cannot be checked")


def _ignored_by_rules(paths: Sequence[str]) -> set[str]:
    """Return the subset of ``paths`` the ignore rules match, index aside."""
    completed = _git("check-ignore", "--no-index", *paths)
    # git check-ignore: 0 = at least one path ignored, 1 = none, 128 = error.
    assert completed.returncode in (0, 1), completed.stderr
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def test_secret_surface_paths_are_ignored() -> None:
    missing = sorted(set(SECRET_PATHS) - _ignored_by_rules(SECRET_PATHS))
    assert not missing, (
        "these secret-bearing paths are not ignored by .gitignore and would be "
        "swept in by `git add -A`:\n  " + "\n  ".join(missing)
    )


def test_env_example_is_not_ignored_and_stays_tracked() -> None:
    ignored = _ignored_by_rules([EXAMPLE_PATH])
    assert not ignored, f"{EXAMPLE_PATH} is ignored by the rules; the template must stay visible"

    tracked = _git("ls-files", "--error-unmatch", EXAMPLE_PATH)
    assert tracked.returncode == 0, (
        f"{EXAMPLE_PATH} is not tracked; the example env file must be committed"
    )


def test_no_tracked_file_is_ignored() -> None:
    # Pin core.excludesFile so a developer's global gitignore cannot fail this
    # assertion: the invariant under test is the repository's own rules.
    completed = _git(
        "-c",
        "core.excludesFile=/dev/null",
        "ls-files",
        "-ci",
        "--exclude-standard",
    )
    assert completed.returncode == 0, completed.stderr

    ignored = [line for line in completed.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "these tracked files are ignored by the repository's rules and would "
        "disappear from `git status`:\n  " + "\n  ".join(ignored)
    )
