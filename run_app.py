#!/usr/bin/env python3
"""Launch the NeuroDecode desktop/web app.

One command starts everything — the decoder API and the web UI are served by a
single local server, then your browser opens on it:

    python run_app.py                 # http://127.0.0.1:8000
    python run_app.py --port 9000     # a different port
    python run_app.py --no-browser    # don't auto-open a browser
    python run_app.py --reload        # auto-restart while editing the code

Nothing is exposed to the internet: the server binds to localhost only.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (default: localhost only)")
    p.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    p.add_argument("--no-browser", action="store_true",
                   help="do not open a browser window automatically")
    p.add_argument("--reload", action="store_true",
                   help="auto-reload on code changes (development)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        # Show the real import error: it is usually either a genuinely missing
        # package or — most often on Windows — 'pip' having installed into a
        # different Python than the one running this script.
        sys.exit(
            f"Could not import uvicorn: {exc}\n\n"
            f"This interpreter is:\n  {sys.executable}\n\n"
            "Install the dependencies into *this* interpreter with:\n"
            "  python -m pip install -r requirements.txt\n"
            "(using 'python -m pip' guarantees pip and python are the same "
            "installation)"
        )

    # uvicorn imports 'main:app' from the backend package directory.
    sys.path.insert(0, BACKEND_DIR)

    url = f"http://{args.host}:{args.port}"
    print(f"\n  NeuroDecode — EEG motor-imagery decoder")
    print(f"  UI      : {url}")
    print(f"  API docs: {url}/docs")
    print("  (Ctrl+C to stop)\n")

    if not args.no_browser:
        # Open the browser shortly after the server starts accepting connections.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=BACKEND_DIR,
        log_level="info",
    )


if __name__ == "__main__":
    main()
