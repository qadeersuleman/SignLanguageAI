"""One-command launcher for SignBridge AI."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn


def _open_browser_later(url: str) -> None:
    time.sleep(1.6)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="SignBridge AI — Sign Language to Screen & Voice Translator")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the web browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 64)
    print("  SignBridge AI — Bridging Communication with AI!")
    print(f"  Open in browser: {url}")
    print("=" * 64)

    if not args.no_browser:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()

    uvicorn.run("app.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
