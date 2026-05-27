"""
main.py — Run the FastAPI server.

    python main.py                  # default port 8000
    python main.py --port 8080      # custom port
"""

import argparse, uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=True)
