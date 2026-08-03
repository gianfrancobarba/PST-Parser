SHELL := /bin/bash
UV ?= uv
COMPOSE ?= docker compose -f docker/compose.yaml

CONFIG ?= configs/experiments/baseline.yaml
ADAPTER ?=
PREDICTIONS ?= results/predictions.jsonl

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-slow check \
        validate-config prepare-data train generate score align synth \
        docker-build-eval docker-build-train docker-score docker-train clean

help: ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Environment -----------------------------------------------------------

install: ## Create the development environment
	$(UV) sync --extra dev --extra cpu --extra train

# --- Quality gates ---------------------------------------------------------

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

# --- Pipeline --------------------------------------------------------------

validate-config: ## Compose and validate CONFIG without running anything
	$(UV) run pstparser validate-config --config $(CONFIG)

prepare-data: ## Convert the annotated corpus into records and partition it
	$(UV) run pstparser prepare-data --config $(CONFIG)

train: ## Fine-tune adapters on the prepared corpus
	$(UV) run pstparser train --config $(CONFIG)

generate: ## Produce predictions; pass ADAPTER=<dir>
	@test -n "$(ADAPTER)" || { echo "set ADAPTER=<directory holding the adapter>"; exit 1; }
	$(UV) run pstparser generate --config $(CONFIG) --adapter $(ADAPTER)

score: ## Compute metrics from PREDICTIONS; needs no accelerator
	$(UV) run pstparser score --config $(CONFIG) --predictions $(PREDICTIONS)

align: ## Locate the phrases of PREDICTIONS in their prompts; needs no accelerator
	$(UV) run pstparser align --config $(CONFIG) --predictions $(PREDICTIONS)

synth: ## Generate prompts for the reasoning paradigms and export them for annotation
	$(UV) run pstparser synth --config $(CONFIG)

# --- Containers ------------------------------------------------------------

docker-build-eval: ## Build the CPU-only scoring image
	docker build -f docker/Dockerfile --target eval -t pstparser:eval .

docker-build-train: ## Build the GPU training image
	docker build -f docker/Dockerfile --target train -t pstparser:train .

docker-score: ## Compute metrics inside the scoring container
	$(COMPOSE) --profile eval run --rm score score \
		--config $(CONFIG) --predictions $(PREDICTIONS)

docker-train: ## Fine-tune inside the training container
	$(COMPOSE) --profile train run --rm train train --config $(CONFIG)

# --- Housekeeping ----------------------------------------------------------

clean: ## Remove caches and generated artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
