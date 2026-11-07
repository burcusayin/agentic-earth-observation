#!/bin/sh
# Pre-commit check: block private files, strings from an optional local denylist, and co-author lines.
#
# Without arguments (pre-commit stage) it checks the staged files, the staged diff and the commit identity.
# With nothing staged, as in CI, it checks every tracked file's name instead.
# With a file argument (commit-msg stage) it checks the commit message in that file.
#
# The denylist lives in .local/denylist.txt, which is gitignored: one string per line, matched as plain text
# and case-insensitively; lines starting with # are comments. Without the file only the other checks run.
set -eu

denylist="$(git rev-parse --show-toplevel)/.local/denylist.txt"
patterns=$(mktemp)
trap 'rm -f "$patterns"' EXIT
if [ -s "$denylist" ]; then
  grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$denylist" > "$patterns" || true
fi

if [ $# -ge 1 ] && [ -f "$1" ]; then
  message=$(grep -v '^#' "$1" || true)
  if printf '%s\n' "$message" | grep -i -q '^co-authored-by:'; then
    echo "private-strings: the commit message has a Co-authored-by line; commits list the maintainer only." >&2
    exit 1
  fi
  if [ -s "$patterns" ] && printf '%s\n' "$message" | grep -i -F -q -f "$patterns"; then
    echo "private-strings: the commit message contains a string listed in .local/denylist.txt." >&2
    exit 1
  fi
  exit 0
fi

private_paths='(^|/)(\.env(\..+)?|\.envrc|CLAUDE\.local\.md)$|(^|/)(\.local|secrets)/|\.(key|pem)$'
if git diff --cached --quiet; then
  candidates=$(git ls-files)
else
  candidates=$(git diff --cached --name-only --diff-filter=ACMR)
fi
if printf '%s\n' "$candidates" | grep -E -v '(^|/)\.env\.example$' | grep -E -q "$private_paths"; then
  echo "private-strings: a private file is staged or tracked (.env*, .envrc, .local/, CLAUDE.local.md, *.key, *.pem, secrets/)." >&2
  exit 1
fi

[ -s "$patterns" ] || exit 0
if git diff --cached -U0 | grep '^+' | grep -v '^+++' | grep -i -F -q -f "$patterns"; then
  echo "private-strings: the staged diff contains a string listed in .local/denylist.txt." >&2
  exit 1
fi
if { git var GIT_AUTHOR_IDENT; git var GIT_COMMITTER_IDENT; } | grep -i -F -q -f "$patterns"; then
  echo "private-strings: the commit identity uses an address listed in .local/denylist.txt." >&2
  exit 1
fi
