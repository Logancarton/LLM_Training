"""Start Sonara: the heartbeat loop plus the control panel.

    python3 adaptive_loop/run.py              # from the LLM_Training folder
    python3 adaptive_loop/run.py --no-browser
"""
from __future__ import annotations

import sys
import webbrowser

import config
from brain import Brain
from runtime import Control, Runtime
from server import serve


def main() -> None:
    brain = Brain()
    runtime = Runtime(brain, Control())
    runtime.start()
    httpd = serve(runtime)
    url = f"http://{config.PANEL_HOST}:{config.PANEL_PORT}"
    snap = runtime.status()
    print(f"Sonara control panel: {url}")
    print(f"  state: {snap['state']}  |  sensor: {snap['providers']['sensor']}  |  "
          f"generator: {snap['providers']['generator']}  |  embedder: {snap['embedder']}  |  "
          f"evidence records: {snap['counts']['evidence']}")
    if snap["providers"]["generator"] == "mock":
        print("  Ollama is not running, so replies are placeholders. Open Ollama, then restart.")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping (the On/Off flag is unchanged).")
    finally:
        runtime.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
