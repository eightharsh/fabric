# Contributing

Thanks for your interest in the project. This is a small research + college
codebase; the bar is "keep it reproducible and green".

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime + test + lint deps
```

The dataset and fitted checkpoints are **not** in the repo (they're large and
gitignored). See the README to download MVTec AD and train a checkpoint.

## Before you push

```bash
make lint      # ruff check .
make test      # pytest
```

Both must pass. CI (`.github/workflows/ci.yml`) runs the same two commands on
every push and pull request.

## Style

- Lint with `ruff` (config in `pyproject.toml`). The codebase is hand-formatted
  in a compact style, so we lint but do **not** enforce `ruff format`.
- Keep hyperparameters in `config/default.yaml` — it's the single source of
  truth the paper's numbers depend on. Add a CLI flag that defaults to the
  config value rather than hard-coding.
- Add a test when you add behavior. Tests live in `tests/` and must not require
  the dataset or a GPU (mock/synthesize small inputs, as the existing ones do).

## Commit messages

Short imperative subject ("Add AITEX loader"), body explaining the *why* when it
isn't obvious.
