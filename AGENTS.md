# Multi-Agent Git Safety Rules (Backend Repo)

This backend repo is edited by multiple AI agents across multiple IDEs.
Follow these rules before making any edits.

## Required startup checks

1. Run `git status --short` in:
   - this backend repo
   - root repo (`../..`)
   - sibling frontend repo (`../frontend`)
2. Report the output before making changes.
3. If unexpected file changes appear during work, stop and ask for guidance.

## Scope and safety

1. Only edit backend files related to the requested task.
2. Never revert unrelated changes.
3. Never run destructive commands unless explicitly approved:
   - `git reset --hard`
   - `git clean -fd`
   - force push
   - history rewrite/rebase for shared branches
4. Do not stash-pop or auto-apply stash entries unless explicitly requested.

## Branch and commit hygiene

1. If repo is dirty, create a task branch before editing:
   - `agent/<task>-<date>`
2. Keep commits small and atomic.
3. Commit only files for the current backend task.
4. After each commit, show:
   - `git show --name-only --oneline -1`
   - `git status --short`

