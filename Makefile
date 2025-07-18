.PHONY: install precommit test-cpu test-gpu test-all

install:
	pip install -e ".[dev]"

precommit:
	python -m ruff format actors/ tests/ examples/
	python -m ruff check --fix actors/ tests/ examples/
# 	python -m mypy actors/ --ignore-missing-imports
	@echo "✅ All checks passed! Ready to commit! 🎉"

test-cpu:
	python -m pytest tests/ -v -m "cpu or not gpu"

test-gpu:
	python -m pytest tests/ -v -m "gpu"

test-all:
	python -m pytest tests/ -v
