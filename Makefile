.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help setup lint format format-check typecheck test check hooks clean

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

typecheck: ## Type-check src and tests with mypy --strict
	$(UV) run mypy

test: ## Run unit and contract tests with coverage (no network, no models)
	$(UV) run pytest --cov

check: lint format-check typecheck test ## Everything CI runs; must pass before a task is called done

hooks: ## Run every pre-commit hook on all files
	$(UV) run pre-commit run --all-files

clean: ## Remove caches and coverage output
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
