# =============================
#  Minimal Makefile for uv-based Python Project
# =============================

UV ?= uv

.DEFAULT_GOAL := help

.PHONY := venv sync dev hooks lint format pre_commit run clean help

# -----------------------------
#  Environment
# -----------------------------

# Create in-project virtual environment (.venv)
venv:
	$(UV) venv --project .

# Sync dependencies (install/update/remove)
sync:
	$(UV) sync

# Install development dependencies (ruff / black / isort / pre-commit)
dev:
	$(UV) sync --group dev

# Install pre-commit hooks
hooks:
	$(UV) run pre-commit install

# -----------------------------
#  Code Quality
# -----------------------------

# Static analysis without auto-fixing
lint:
	$(UV) run ruff check .

# Auto-format code (isort -> black keeps styles consistent)
format:
	$(UV) run isort . --profile black
	$(UV) run black .
	$(UV) run ruff check . --fix

# -----------------------------
#  Utilities
# -----------------------------

# Run all pre-commit hooks manually
pre_commit:
	$(UV) run pre-commit run --files $$files


# Run the full simulation suite
run:
	$(UV) run python scripts/run_all.py

# Clean caches and venv
clean:
	rm -rf .venv .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# -----------------------------
#  Help Message
# -----------------------------

help:
	@echo "------------------ Available Commands ------------------"
	@echo "make venv        - Create in-project virtual environment"
	@echo "make sync        - Sync dependencies from pyproject/uv.lock"
	@echo "make dev         - Install development dependencies (dev group)"
	@echo "make hooks       - Install pre-commit hooks"
	@echo "make format      - Auto-format code with isort/black/ruff"
	@echo "make lint        - Run Ruff in check-only mode"
	@echo "make pre_commit  - Run all pre-commit hooks"
	@echo "make run         - Execute scripts/run_all.py via uv"
	@echo "make clean       - Remove caches and virtual environment"
	@echo "--------------------------------------------------------"
