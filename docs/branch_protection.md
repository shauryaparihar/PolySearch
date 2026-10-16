# Branch protection (instructions only)

Nothing in this repository changes GitHub settings. These are the steps to apply by hand, in the
repository's **Settings → Branches → Add branch ruleset** (or classic *Branch protection rule*),
for the `main` branch:

1. **Require a pull request before merging** (at least 1 approval if there is a second reviewer;
   for a solo project, still use pull requests so CI runs before the merge).
2. **Require status checks to pass before merging**, and select the three CI jobs by name:
   `backend`, `frontend` and `containers`. Tick *Require branches to be up to date*.
3. **Block force pushes** and **restrict deletions** on `main`.
4. Optional: *Require linear history* and *Require conversation resolution*.

The CI workflow runs on every pull request and every push to `main`, so the checks above exist
as soon as one run has completed. Until the rule is applied, the same discipline holds by
convention: work on a branch, run the CI commands locally (listed in `docs/containers.md`), and
merge only when they pass.

To apply it with the GitHub CLI instead (review the JSON first; this changes remote settings):

```bash
gh api -X PUT repos/<owner>/<repo>/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": {"strict": true, "contexts": ["backend", "frontend", "containers"]},
  "enforce_admins": false,
  "required_pull_request_reviews": {"required_approving_review_count": 0},
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```
