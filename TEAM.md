# Team

The repository is public. Write access is granted per person; a personal GitHub
account has **collaborators**, not organisation teams.

## Members

| Member | GitHub | Repository role | Code owner | Can do |
|---|---|---|---|---|
| Resul | `@resulbegmyradow33-sys` | Admin | yes | everything, including settings, releases, access |
| Lukman | `@lukmanhekim099` | **Write** | yes | branches, PRs, issues, reviews, CI runs; no repo settings |

Write access is deliberately not maintain/admin: it is enough for daily
development, while branch protection, secrets, access and repository settings
stay with the owner.

## What each role can do

**Write (Lukman)**

- create/push topic branches;
- open, review and comment on pull requests;
- create, assign and close issues;
- run GitHub Actions on branches and read logs;
- view security alerts and CodeQL results;
- be assigned issues and milestones.

**Owner only (Resul)**

- merge to `main` when the review/CI gate is not met (admin bypass);
- change branch protection, actions permissions, secrets, collaborators;
- delete/transfer the repository;
- create/change labels, milestones and repository settings;
- publish releases if write access is insufficient for a specific action.

## Review rules

- A PR needs **one approval** from the other member plus green CI
  (`backend`, `frontend`, CodeQL `analyze` jobs).
- Both members are code owners, so either can supply the code-owner review.
  Authors never approve their own PR.
- Conversations must be resolved; the branch must be up to date; squash merge.
- If the owner must bypass the gate for maintenance, the reason goes in the PR
  or issue thread.

## Work assignment

- Issues are the unit of work; pick one from a phase milestone.
- Comment "bende" (or assign yourself) before starting.
- Keep at most three issues in progress per person.
- The board columns are Backlog / Ready / In progress / Review / Done.

## Access hygiene

- Every member uses 2FA on GitHub.
- No shared accounts or tokens.
- Access is reviewed when a member stops contributing.
- Security reports go through `SECURITY.md`, never public issues.

## Organisation option

If the project later needs real GitHub Teams, create a free organisation,
transfer the repository, and create a `core` team with **Write** on the repo.
That changes the repository URL; decide before external links multiply.
