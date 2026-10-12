# Common tasks. Override variables on the command line, e.g.
#   make eval PROMPT=prompts/v3.yaml
#   make baseline PROMPT=prompts/v1.yaml

PYTHON   ?= .venv/bin/python
PROMPT   ?= prompts/v2.yaml
BASELINE_PROMPT ?= prompts/v1.yaml
BASELINE_FILE   ?= baselines/baseline.json
IMAGE    ?= regress:latest

.PHONY: help install build-dataset baseline eval dashboard test docker-build docker-run

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Create .venv and install the package with dev + dashboard extras
	python3 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev,dashboard]"

build-dataset: ## Validate data/golden_draft.yaml and write data/golden.json
	$(PYTHON) scripts/build_dataset.py

baseline: ## Evaluate BASELINE_PROMPT, store it, and export the committed baseline file
	$(PYTHON) -m regress baseline --prompt $(BASELINE_PROMPT) --export $(BASELINE_FILE)

eval: ## Evaluate PROMPT against the committed baseline (exit 1 on critical regression)
	$(PYTHON) -m regress run --prompt $(PROMPT) --baseline $(BASELINE_FILE)

dashboard: ## Open the Streamlit dashboard
	$(PYTHON) -m streamlit run dashboard/app.py

test: ## Run the test suite (LLM calls are mocked)
	$(PYTHON) -m pytest

docker-build: ## Build the runner image
	docker build -t $(IMAGE) .

docker-run: ## Evaluate PROMPT inside Docker (reads secrets from .env)
	mkdir -p out
	docker run --rm --env-file .env -e GIT_SHA=$$(git rev-parse --short=12 HEAD) \
		-v "$(CURDIR)/out:/data" $(IMAGE) run --prompt $(PROMPT) --baseline $(BASELINE_FILE)
