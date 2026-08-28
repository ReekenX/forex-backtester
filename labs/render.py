"""
Render the 15LS1CC lab to a static HTML file.

    poetry run python labs/render.py

Live-reloading workflow (watchexec re-runs this on every save):

    watchexec -w strategies/15LS1CC -e py,csv -- poetry run python labs/render.py

Open labs/build/15LS1CC.html directly, or serve the directory:

    python3 -m http.server -d labs/build 8000

Either way the page reloads only when the data actually changes, and restores
your column sort and scroll position afterwards.

Pass --no-reload for a frozen snapshot (sharing, printing to PDF, screenshots).
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "strategies" / "15LS1CC"
sys.path.insert(0, str(STRATEGY_DIR))

from utils.confirmation_candle import load_data  # noqa: E402
from utils.report import render_error_to_file, render_to_file  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "labs" / "build" / "15LS1CC.html"


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    live_reload = "--no-reload" not in argv
    out_path = Path(args[0]).resolve() if args else DEFAULT_OUT

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        df = load_data(str(STRATEGY_DIR / "data.csv"))
        build_id = render_to_file(df, out_path, generated_at, live_reload)
    except Exception:
        # A half-written CSV export raises inside pandas. Render the failure so
        # the browser shows it instead of a silently stale page, and keep the
        # watcher alive for the next save.
        message = traceback.format_exc()
        build_id = render_error_to_file(message, out_path, generated_at)
        print(f"{generated_at}  BUILD FAILED  build {build_id}  ->  {out_path}",
              file=sys.stderr)
        print(message, file=sys.stderr)
        return 1

    print(f"{generated_at}  {len(df)} trades  build {build_id}  ->  {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
