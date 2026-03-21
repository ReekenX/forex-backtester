.PHONY: install clean format lint test

install:
	@poetry install --no-root

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

format:
	@black strategies/

lint:
	@ruff check strategies/ --fix

test:
	@poetry run python -m pytest strategies/15LS1CC/tests/ -v
	@poetry run python -m pytest strategies/5OB1CC/tests/ -v