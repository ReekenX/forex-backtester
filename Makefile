.PHONY: install clean format lint test report report-5ob watch watch-5ob serve open open-5ob

# The 15-minute lab. Each strategy renders its own page into labs/build.
REPORT_PAGE := 15C.html
REPORT_PAGE_5OB := 5OB.html

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

report-5ob:
	@poetry run python labs/render_5OB.py

open: report
	@open labs/build/$(REPORT_PAGE)

open-5ob: report-5ob
	@open labs/build/$(REPORT_PAGE_5OB)

watch:
	@command -v watchexec >/dev/null 2>&1 || { echo "watchexec not found - install it with: brew install watchexec"; exit 1; }
	@watchexec -w strategies/15LS1CC -e py,csv -- poetry run python labs/render.py

watch-5ob:
	@command -v watchexec >/dev/null 2>&1 || { echo "watchexec not found - install it with: brew install watchexec"; exit 1; }
	@watchexec -w strategies/5OB1CC -e py,csv -- poetry run python labs/render_5OB.py

serve:
	@echo "http://localhost:8000/$(REPORT_PAGE)"
	@echo "http://localhost:8000/$(REPORT_PAGE_5OB)"
	@python3 -m http.server -d labs/build 8000
