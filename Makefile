SHELL := /bin/bash
CONFIG ?= configs/experiments/baseline.yaml
UV ?= uv

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-slow check validate-config clean

help: ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create the development environment
	$(UV) sync --extra dev

lint: ## Report style and correctness issues
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Apply automatic formatting and safe fixes
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## Verify type annotations
	$(UV) run mypy

test: ## Run the fast test suite
	$(UV) run pytest

test-slow: ## Run the end-to-end recipe tests
	$(UV) run pytest -m slow

check: lint typecheck test ## Run every quality gate

validate-config: ## Compose and validate CONFIG without running anything
	$(UV) run pstparser validate-config --config $(CONFIG)

clean: ## Remove caches and generated artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
