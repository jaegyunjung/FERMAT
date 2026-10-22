# Repository commit identity

For work performed for the repository owner, use this identity for new commits:

- Author and committer name: `jaegyunjung`
- Author and committer email: `jaegyun.jung@snu.ac.kr`

Before committing, verify `git var GIT_AUTHOR_IDENT` and `git var GIT_COMMITTER_IDENT`.
If the environment supplies a different identity, set repository-local `user.name`
and `user.email` and the corresponding `GIT_AUTHOR_*` / `GIT_COMMITTER_*`
environment variables for the commit command to the values above. Do not change
global Git configuration. This also applies in cloud environments that do not
load the `env` block from `.claude/settings.json`.

Do not append Claude co-author trailers, `Generated with Claude Code`,
`Claude-Session` trailers, or Claude session links to commit messages or PR
bodies. Keep legitimate human attribution intact.

Apply this policy only to new work. Do not amend, rebase, filter, or force-push
existing commits merely to change attribution. Existing history must stay intact.
