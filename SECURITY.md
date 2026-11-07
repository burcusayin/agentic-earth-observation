# Security

This is a research project that runs on your own machine. Its servers bind to 127.0.0.1 only, refuse requests
with a foreign Host (and, for the retrieval agent, any Origin), and hold no credentials of their own. They call a
public catalogue (Earth Search by default) and a local model server (Ollama).

## Reporting a problem

Please report a vulnerability privately, through
[GitHub's private reporting](https://github.com/burcusayin/agentic-earth-observation/security/advisories/new),
rather than in a public issue. This is a personal project with no guaranteed response time; I will answer as soon
as I can.

Useful things to report: a way for a catalogue's answer, a tool result or a user's text to make a server act
beyond its tools, to reach another service, or to get a fabricated scene into an answer.

Only the latest commit on `main` is looked after.
