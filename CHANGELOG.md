# Changelog

All notable changes to this project. Format based on Keep a Changelog; the
project does not yet follow strict semantic versioning.

## Unreleased

### Changed

- Repository made public. Server-side `main` protection, secret scanning,
  push protection and CodeQL are now enabled at no cost.

### Added

- Symbol registry for ten instruments and M30/H4 timeframes.
- Bar/tick validation reports and dataset provenance sidecars.
- JSON bar import and `data derive` CLI.
- API endpoints for symbols, data validation and provenance.
- Feature-group registry covering all 59 causal columns.
- Chronological dataset builder with purge/embargo and manifest checksums.
- Team infrastructure: CI, PR/issue templates, CODEOWNERS, security policy,
  contributor guide, team workflow, labels, milestones and GitHub setup script.
- Local `pre-push` guard for the private GitHub Free plan, plus `make hooks`.
- Calmer Dependabot policy: major versions ignored, minor/patch grouped.

## 2026-11-22

### Added

- Master requirements brief, architecture audit, implementation roadmap and
  one-page status matrix.
- Forward test on unseen June–November 2026 ticks: 24 trades, +0.502 R,
  PF 2.15, +$481 (small sample; confidence interval includes zero).
- Ablation showing momentum without the signed-flow gate loses money.
- Weekly forward update job and history.
- Behavior-cloning warm start for PPO and a data-gated RL lab.
- Live strategy card showing its holdout and latest forward result.

### Changed

- The default branch is `main` (renamed from `feat/v1`).

## Earlier

- Initial backend: Parquet store, 59 causal features, paper broker, risk engine,
  Gymnasium simulator, PPO, walk-forward evaluation, cost stress, FastAPI and
  the Next.js panel.
- Validated `flow_momentum` rule and nightly calibration with a holdout gate.
