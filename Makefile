.PHONY: install clean format lint test report watch serve

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

report:
	@poetry run python labs/render.py

watch:
	@command -v watchexec >/dev/null 2>&1 || { echo "watchexec not found - install it with: brew install watchexec"; exit 1; }
	@watchexec -w strategies/15LS1CC -e py,csv -- poetry run python labs/render.py

serve:
	@python3 -m http.server -d labs/build 8000
