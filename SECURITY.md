# Security policy

## Supported versions

Only the `main` branch is supported. Experiment branches are not.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:

https://github.com/resulbegmyradow33-sys/ai-scalping-studio/security/advisories/new

If that form is unavailable on the current plan, open a regular issue titled
`security contact request` **without details** and the maintainer will provide
a private channel.

Do not open a public issue with the vulnerability details. Include the commit,
a minimal reproduction, impact and any suggested fix. The maintainer aims to
acknowledge within 7 days and to publish a fix or mitigation before disclosure.

## Scope

In scope:

- the FastAPI backend and WebSocket runner;
- data ingestion and provenance;
- model training, serialization and loading;
- the Next.js frontend;
- GitHub Actions and repository configuration.

Out of scope by design:

- real-money execution (the project has no enabled broker adapter);
- the accuracy of third-party market data;
- trading losses from research strategies.

## Secrets

No API keys, broker credentials, `.env` files or private data are committed.
If a secret is committed, rotate it first, then remove it from history and
notify the maintainer. GitHub secret scanning push protection should be
enabled on the repository.
