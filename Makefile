# Convenience targets. Uses the local .venv if present, else system python.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
CATEGORY ?= carpet

.PHONY: help install install-dev test lint format-check api web docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install runtime + dev/test dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check .

format-check:  ## Report formatting drift (does not modify files)
	$(PYTHON) -m ruff format --check .

api:  ## Run the inference API (FD_CATEGORY=$(CATEGORY))
	FD_CATEGORY=$(CATEGORY) $(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

web:  ## Serve the frontend on :5173
	cd frontend/web && $(PYTHON) -m http.server 5173

docker:  ## Build the backend image
	docker build -t fabric-defect .

clean:  ## Remove caches and __pycache__
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
