#!/bin/sh
# Pre-commit check: block private files, and block strings listed in an optional local denylist.
#
# The denylist lives in .local/denylist.txt, which is gitignored: one string per line, matched as plain
# text and case-insensitively; lines starting with # are comments. Without the file only the path check runs.
set -eu

private_paths='(^|/)(\.env(\..+)?|\.envrc|CLAUDE\.local\.md)$|(^|/)(\.local|secrets)/|\.(key|pem)$'
if git diff --cached --name-only --diff-filter=ACMR | grep -E -v '(^|/)\.env\.example$' | grep -E -q "$private_paths"; then
  echo "private-strings: a private file is staged (.env*, .envrc, .local/, CLAUDE.local.md, *.key, *.pem, secrets/)." >&2
  exit 1
fi

denylist="$(git rev-parse --show-toplevel)/.local/denylist.txt"
[ -s "$denylist" ] || exit 0

patterns=$(mktemp)
trap 'rm -f "$patterns"' EXIT
grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$denylist" > "$patterns" || true
[ -s "$patterns" ] || exit 0

if git diff --cached -U0 | grep '^+' | grep -v '^+++' | grep -i -F -q -f "$patterns"; then
  echo "private-strings: the staged diff contains a string listed in .local/denylist.txt." >&2
  exit 1
fi
if { git var GIT_AUTHOR_IDENT; git var GIT_COMMITTER_IDENT; } | grep -i -F -q -f "$patterns"; then
  echo "private-strings: the commit identity uses an address listed in .local/denylist.txt." >&2
  exit 1
fi
