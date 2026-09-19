.PHONY: install clean format lint test \
	report report-15c report-5ob \
	watch watch-15c watch-5ob \
	open open-15c open-5ob serve

# Each strategy renders its own page into labs/build. The bare report / watch /
# open targets follow the strategy under active work - 30MCR - and every page
# keeps an explicit target of its own.
REPORT_PAGE := 30MCR.html
REPORT_PAGE_15C := 15C.html
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
	@poetry run python -m pytest strategies/30MCR/tests/ -v
	@poetry run python -m pytest strategies/15LS1CC/tests/ -v
	@poetry run python -m pytest strategies/5OB1CC/tests/ -v

report:
	@poetry run python labs/render_30MCR.py

report-15c:
	@poetry run python labs/render.py

report-5ob:
	@poetry run python labs/render_5OB.py

open: report
	@open labs/build/$(REPORT_PAGE)

open-15c: report-15c
	@open labs/build/$(REPORT_PAGE_15C)

open-5ob: report-5ob
	@open labs/build/$(REPORT_PAGE_5OB)

# watchexec is event-driven (OS filesystem notifications), not a polling timer.
define require_watchexec
@command -v watchexec >/dev/null 2>&1 || { echo "watchexec not found - install it with: brew install watchexec"; exit 1; }
endef

watch:
	$(require_watchexec)
	@watchexec -w strategies/30MCR -e py,csv -- poetry run python labs/render_30MCR.py

watch-15c:
	$(require_watchexec)
	@watchexec -w strategies/15LS1CC -e py,csv -- poetry run python labs/render.py

watch-5ob:
	$(require_watchexec)
	@watchexec -w strategies/5OB1CC -e py,csv -- poetry run python labs/render_5OB.py

serve:
	@echo "http://localhost:8000/$(REPORT_PAGE)"
	@echo "http://localhost:8000/$(REPORT_PAGE_15C)"
	@echo "http://localhost:8000/$(REPORT_PAGE_5OB)"
	@python3 -m http.server -d labs/build 8000
