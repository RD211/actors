.PHONY: install precommit test-cpu test-gpu test-all

precommit:
	python -m ruff format actors/ tests/ examples/
	python -m ruff check --fix actors/ tests/ examples/
# 	python -m mypy actors/ --ignore-missing-imports
	@echo "✅ All checks passed! Ready to commit! 🎉"

install:
	pip install -e ".[dev]"

test:
	ACTORS_LOGGING_LEVEL='silent' python -m pytest tests/ -v -m "not slow"

test-slow:
	ACTORS_LOGGING_LEVEL='silent' python -m pytest tests/ -v -m "slow"

test-all:
	ACTORS_LOGGING_LEVEL='silent' python -m pytest tests/ -v -m "not slow or slow"
