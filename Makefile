.PHONY: install clean format lint test \
	report report-15c report-5ob report-30mcr-fixed \
	watch watch-15c watch-5ob watch-30mcr-fixed \
	open open-15c open-5ob open-30mcr-fixed serve

# Each strategy renders its own page into labs/build. The bare report / watch /
# open targets follow the strategy under active work - 30MCR - and every page
# keeps an explicit target of its own.
REPORT_PAGE := 30MCR.html
REPORT_PAGE_15C := 15C.html
REPORT_PAGE_5OB := 5OB.html
REPORT_PAGE_30MCR_FIXED := 30MCR-fixed.html

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
	@poetry run python -m pytest strategies/30MCR-fixed/tests/ -v
	@poetry run python -m pytest strategies/15LS1CC/tests/ -v
	@poetry run python -m pytest strategies/5OB1CC/tests/ -v

report:
	@poetry run python labs/render_30MCR_fixed.py

report-15c:
	@poetry run python labs/render.py

report-5ob:
	@poetry run python labs/render_5OB.py

report-30mcr:
	@poetry run python labs/render_30MCR.py

open: report
	@open labs/build/$(REPORT_PAGE)

open-15c: report-15c
	@open labs/build/$(REPORT_PAGE_15C)

open-5ob: report-5ob
	@open labs/build/$(REPORT_PAGE_5OB)

open-30mcr-fixed: report-30mcr-fixed
	@open labs/build/$(REPORT_PAGE_30MCR_FIXED)

# watchexec is event-driven (OS filesystem notifications), not a polling timer.
define require_watchexec
@command -v watchexec >/dev/null 2>&1 || { echo "watchexec not found - install it with: brew install watchexec"; exit 1; }
endef

watch:
	$(require_watchexec)
	@watchexec -w strategies/30MCR-fixed -e py,csv -- poetry run python labs/render_30MCR_fixed.py

watch-15c:
	$(require_watchexec)
	@watchexec -w strategies/15LS1CC -e py,csv -- poetry run python labs/render.py

watch-5ob:
	$(require_watchexec)
	@watchexec -w strategies/5OB1CC -e py,csv -- poetry run python labs/render_5OB.py

watch-30mcr-fixed:
	$(require_watchexec)
	@watchexec -w strategies/30MCR-fixed -e py,csv -- poetry run python labs/render_30MCR_fixed.py

serve:
	@echo "http://localhost:8000/$(REPORT_PAGE)"
	@echo "http://localhost:8000/$(REPORT_PAGE_15C)"
	@echo "http://localhost:8000/$(REPORT_PAGE_5OB)"
	@echo "http://localhost:8000/$(REPORT_PAGE_30MCR_FIXED)"
	@python3 -m http.server -d labs/build 8000
