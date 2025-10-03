.PHONY: run install clean format lint

run:
	@poetry run jupyter notebook main.ipynb

install:
	@poetry install --no-root

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

format:
	@black .

lint:
	@ruff check . --fix