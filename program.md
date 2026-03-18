# autoresearch-saas

This repository now runs in AutoSaaS mode.

## Scope

- Build new product behavior in `autosaas/`.
- Treat `legacy/` as read-only historical context.
- Keep migrations incremental and test-backed.

## Operator checklist

1. Confirm `autosaas/` and `legacy/` both exist.
2. Run focused tests before and after each change.
3. Avoid editing legacy runtime files unless explicitly requested.
