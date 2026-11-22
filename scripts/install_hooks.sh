#!/usr/bin/env bash
# Point this clone/worktree at the repository's hooks.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
git -C "$root" config core.hooksPath .githooks
chmod +x "$root/.githooks/"* 2>/dev/null || true
echo "installed git hooks from $root/.githooks"
