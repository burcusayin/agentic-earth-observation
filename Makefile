.DEFAULT_GOAL := help
UV ?= uv

INSPECTOR_VERSION := 2.12.0

.PHONY: help setup lint format format-check typecheck test check hooks clean serve-mcp serve-agent inspector

help: ## List the available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: ## Create .venv from uv.lock with uv's own Python, and install the git hooks
	$(UV) sync --locked --managed-python
	$(UV) run pre-commit install --overwrite

lint: ## Lint with ruff
	$(UV) run ruff check .

format: ## Format with ruff
	$(UV) run ruff format .

format-check: ## Fail if any file needs formatting
	$(UV) run ruff format --check .

typecheck: ## Type-check src, tests and scripts with mypy --strict
	$(UV) run mypy

test: ## Run unit and contract tests with coverage (no network, no models)
	$(UV) run pytest --cov

check: lint format-check typecheck test ## Everything CI runs; must pass before a task is called done

hooks: ## Run every pre-commit hook on all files
	$(UV) run pre-commit run --all-files

serve-mcp: ## Run the catalogue MCP server on http://127.0.0.1:8001/mcp (Ctrl+C to stop)
	$(UV) run python -m agentic_earth_observation.catalog_mcp.server

serve-agent: ## Run the retrieval agent's A2A server on http://127.0.0.1:8002 (needs make serve-mcp and Ollama)
	$(UV) run python -m agentic_earth_observation.retrieval_agent.a2a_server

inspector: ## Open the MCP Inspector on the running catalogue server (needs Node; start make serve-mcp first)
	npx -y @modelcontextprotocol/inspector@$(INSPECTOR_VERSION) --server-url http://127.0.0.1:8001/mcp --transport http

clean: ## Remove caches and coverage output
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
	find src tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +
