# eo-agent-demo: rules for Claude Code

See README.md for the plan and milestones; `docs/decisions.md` for decisions already made.

- Type annotations on every function signature.
- Validate everything that crosses a boundary: tool arguments, model outputs, STAC responses, file reads.
- The agent may only cite scene ids that a tool returned in the same conversation.
- Keep the golden set and raw results in git so every number in the summary can be recomputed
  (except `results/raw/`, which is ignored; summaries are committed).
- Secrets live in `.env` only. `.env.example` lists the variables with empty values. Never commit a key.
- Private notes live in `.local/` and `CLAUDE.local.md`, both gitignored. If present, read them first.
- Prefer the smallest model that passes the evaluation; record cost per task in the results.
