# 🔐 Security & Secrets Handling

- **Never commit real secrets** (API keys, tokens, passwords). Commit only placeholder values in `.env.example`.
- Add `.env*` to `.gitignore` (see snippet below).
- Prefer environment variables over hardcoding secrets.
- For production, use a secret manager (e.g., Vault, AWS Secrets Manager, Doppler, Kubernetes Secrets).

## .gitignore snippet

```
# Env files
.env
.env.*
!.env.example
```

## Rotating Credentials
- Rotate tokens regularly.
- Scope tokens minimally (read-only where possible).
- Remove unused credentials promptly.