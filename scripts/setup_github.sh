#!/usr/bin/env bash
# Apply the team settings that the repository owner can change with gh.
# Idempotent: safe to run again after a settings drift.
#
# Usage:
#   ./scripts/setup_github.sh
#   REPO=owner/name ./scripts/setup_github.sh
set -euo pipefail

REPO="${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
echo "Configuring $REPO"

if ! gh api "repos/$REPO/branches/main" >/dev/null 2>&1; then
  echo "error: branch 'main' does not exist yet; push it first" >&2
  exit 1
fi

# --- repository basics -------------------------------------------------------
gh repo edit "$REPO" \
  --description "Local, zero-cost XAUUSD research and paper-trading studio: causal features, realistic simulator, PPO and a validated flow rule." \
  --enable-issues \
  --enable-projects \
  --enable-discussions \
  --enable-squash-merge \
  --enable-merge-commit=false \
  --enable-rebase-merge=false \
  --enable-auto-merge \
  --delete-branch-on-merge \
  --squash-merge-commit-message pr-title \
  --default-branch main

for topic in algorithmic-trading quantitative-finance reinforcement-learning \
  machine-learning xauusd pytorch fastapi nextjs paper-trading research; do
  gh repo edit "$REPO" --add-topic "$topic" >/dev/null 2>&1 || true
done

# --- labels ------------------------------------------------------------------
label() {
  gh label create "$1" --repo "$REPO" --color "$2" --description "$3" --force >/dev/null
}

label "bug" "d73a4a" "Something is broken or behaves differently than documented"
label "enhancement" "a2eeef" "New feature or improvement"
label "data" "0e8a16" "Market data import, validation or provenance"
label "experiment" "5319e7" "Model training/evaluation experiment"
label "research" "1d76db" "Exploratory research question"
label "docs" "0075ca" "Documentation"
label "infra" "6e7781" "CI, repository settings, tooling"
label "security" "b60205" "Security issue; report privately"
label "triage" "ededed" "Needs maintainer triage"
label "blocked" "000000" "Blocked by another issue or dependency"
label "good first issue" "7057ff" "Scoped for a new contributor"
label "help wanted" "008672" "Maintainer would welcome help"
label "priority:high" "e11d21" "Next milestone"
label "priority:normal" "fbca04" "Planned"
label "priority:low" "c2e0c6" "Nice to have"
label "promotion" "0e8a16" "Candidate model promoted to champion"
label "rollback" "b60205" "Champion rollback"

# --- milestones --------------------------------------------------------------
existing="$(gh api "repos/$REPO/milestones?state=all&per_page=100" --paginate --jq '.[].title')"
for phase in \
  "Phase 2: data foundation" \
  "Phase 3: features and datasets" \
  "Phase 4: regime engine" \
  "Phase 5: classical ML" \
  "Phase 6: neural networks" \
  "Phase 7: transformers" \
  "Phase 8: RL expansion" \
  "Phase 9: ensemble and confidence" \
  "Phase 10: autonomous learning" \
  "Phase 11: risk and backtesting" \
  "Phase 12: paper trading" \
  "Phase 13: dashboard" \
  "Phase 14: optimisation" \
  "Phase 15: acceptance"; do
  if ! grep -qxF "$phase" <<<"$existing"; then
    gh api -X POST "repos/$REPO/milestones" -f title="$phase" -f state="open" >/dev/null
    echo "created milestone: $phase"
  fi
done

# --- branch protection -------------------------------------------------------
# Classic protection is used because it is supported on private repositories
# without Advanced Security. If the account plan rejects it, the script warns
# and the maintainer can apply the same rules in Settings → Branches.
if gh api --method PUT "repos/$REPO/branches/main/protection" --input - >/dev/null <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["backend", "frontend"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 1,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
JSON
then
  echo "branch protection applied to main"
else
  echo "warning: branch protection was rejected (plan or permissions); configure it in Settings → Branches" >&2
fi

# --- security ----------------------------------------------------------------
gh api --method PUT "repos/$REPO/vulnerability-alerts" >/dev/null 2>&1 \
  && echo "vulnerability alerts enabled" \
  || echo "warning: vulnerability alerts not enabled (plan or permissions)"

gh api --method PUT "repos/$REPO/automated-security-fixes" >/dev/null 2>&1 \
  && echo "Dependabot security fixes enabled" \
  || echo "warning: Dependabot security fixes not enabled (plan or permissions)"

echo "done: https://github.com/$REPO"
