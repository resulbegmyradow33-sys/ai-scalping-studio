"""Rewrite the committed public API snapshot from the live FastAPI app.

The same file is written by ``UPDATE_OPENAPI=1 uv run pytest
tests/test_openapi_snapshot.py``; this script is the non-pytest path for a
maintainer who wants the snapshot without running the suite.

    cd backend
    uv run python scripts/update_openapi_snapshot.py
    git diff tests/snapshots/openapi.json

Exit codes: 0 when the snapshot was written, 2 when the app could not produce a
document. Preview a diff without touching the file with ``--check``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from scalping.api.app import create_app
from scalping.api.openapi_contract import (
    DEFAULT_SNAPSHOT_PATH,
    build_snapshot,
    diff_snapshots,
    write_snapshot,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_SNAPSHOT_PATH),
        help="snapshot path (default: backend/tests/snapshots/openapi.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print the drift instead of writing; exit 1 when it differs",
    )
    args = parser.parse_args(argv)

    try:
        # ``create_app`` imports the modules and builds the models; it opens no
        # store and starts no thread, so this is safe outside the API process.
        snapshot = build_snapshot(create_app().openapi())
    except Exception as exc:  # noqa: BLE001 - script boundary; any failure is exit 2
        print(f"cannot build the OpenAPI snapshot: {exc}", file=sys.stderr)
        return 2

    target = Path(args.out)
    if args.check:
        if not target.exists():
            print(f"missing snapshot {target}", file=sys.stderr)
            return 1
        committed = json.loads(target.read_text(encoding="utf-8"))
        drift = diff_snapshots(committed, snapshot)
        if not drift:
            print(f"snapshot is current: {target}")
            return 0
        print(drift)
        return 1

    write_snapshot(target, snapshot)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
