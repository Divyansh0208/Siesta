"""Tiny local web server (standard library only). Listens on 127.0.0.1 and nothing else."""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import draft, engine, plan, sources, store

WEB = Path(__file__).parent / "web" / "index.html"
MAX_BODY = 25 * 1024 * 1024


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.df = self.meta = self.model = None
        self.model_error = None
        self.job = {"state": "idle", "log": [], "error": None}

    def set_history(self, df, meta, save=True):
        self.df, self.meta, self.model, self.model_error = df, meta, None, None
        try:
            self.model = engine.fit(df)
        except engine.HistoryError as e:
            self.model_error = str(e)
        if save:
            store.save_history(df, meta)

    def state(self) -> dict:
        installed = draft.models()
        s = {"history": None, "report": None, "heat": None, "error": self.model_error,
             "llm": {"up": installed is not None, "models": installed or [],
                        "default": draft.default_model(installed or [])},
             "job": self.job, "plan": store.latest_plan()}
        if self.df is not None:
            s["history"] = {**self.meta, "rows": len(self.df),
                            "from": str(self.df["when"].iloc[0].date()),
                            "to": str(self.df["when"].iloc[-1].date())}
        if self.model:
            s["report"], s["heat"] = self.model.report, self.model.heat
        return s

    def start(self, body: dict):
        brief = str(body.get("brief", "")).strip()[:2000]
        n = min(7, max(1, int(body.get("n", 5))))
        model = body.get("model") or None
        if not brief:
            raise ValueError("Write a short brief first: what should the posts be about?")
        if self.model is None:
            raise ValueError(self.model_error or "Add a post history first.")
        if model and model not in (draft.models() or []):
            raise ValueError(f"'{model}' is not available to your Gemini key.")
        with self.lock:
            if self.job["state"] == "running":
                raise RuntimeError("A run is already in progress.")
            self.job = {"state": "running", "log": [], "error": None}
        threading.Thread(target=self._run, args=(brief, n, model), daemon=True).start()

    def _run(self, brief, n, model):
        def log(msg):
            self.job["log"].append(msg)
        try:
            log("Starting")
            p = plan.make_plan(self.df, self.model, brief, n, model, log=log)
            store.save_plan(p)
            self.job["state"] = "done"
        except Exception as e:  # shown to the person; the log keeps the rest
            self.job.update(state="error", error=str(e))


APP = App()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, payload, ctype="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("That file is too large (25 MB limit).")
        return self.rfile.read(n)

    def _json(self) -> dict:
        raw = self._body()
        return json.loads(raw) if raw else {}

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                self._send(200, WEB.read_bytes(), "text/html")
            elif u.path == "/api/state":
                self._send(200, APP.state())
            elif u.path == "/api/schedule":
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                self._send(200, plan.schedule_text(q.get("day", "mon"), q.get("time", "02:00"),
                                                   q.get("source", "/path/to/your/project"),
                                                   q.get("kind", "git"), sys.executable,
                                                   str(Path.cwd())))
            else:
                self._send(404, {"error": "Not found"})
        except ValueError as e:
            self._send(400, {"error": str(e)})

    def do_POST(self):
        try:
            p = urlparse(self.path).path
            if p == "/api/history/csv":
                df = sources.from_csv_text(self._body().decode("utf-8", "replace"))
                APP.set_history(df, {"kind": "csv", "label": "Your CSV"})
            elif p == "/api/history/dev":
                b = self._json()
                df = sources.fetch_dev(b.get("username", ""), b.get("tag", ""))
                who = f"@{b['username'].strip().lstrip('@')}" if b.get("username") else f"#{b.get('tag')}"
                APP.set_history(df, {"kind": "dev", "label": f"DEV posts, {who}"})
            elif p == "/api/history/demo":
                APP.set_history(sources.demo_history(), {"kind": "demo", "label": "Demo data (made up)"})
            elif p == "/api/brief":
                b = self._json()
                text = (sources.brief_from_git(b["path"]) if b.get("kind") == "git"
                        else sources.brief_from_file(b["path"]))
                return self._send(200, {"brief": text})
            elif p == "/api/run":
                APP.start(self._json())
            elif p == "/api/score":
                if APP.model is None:
                    raise ValueError("Add a post history first.")
                d = self._json()["draft"]
                d = {"title": str(d["title"])[:150], "tags": d["tags"], "minutes": float(d["minutes"]),
                     "cover": int(bool(d["cover"])), "wd": int(d["wd"]) % 7, "hour": int(d["hour"]) % 24}
                return self._send(200, engine.score_draft(APP.model, d))
            else:
                return self._send(404, {"error": "Not found"})
            self._send(200, {"ok": True})
        except (ValueError, KeyError, TypeError, engine.HistoryError) as e:
            self._send(400, {"error": str(e) or "That did not look right."})
        except RuntimeError as e:
            self._send(409, {"error": str(e)})


def serve(port: int = 8765, open_browser: bool = True) -> None:
    loaded = store.load_history()
    if loaded:
        try:
            APP.set_history(*loaded, save=False)
        except engine.HistoryError as e:
            APP.model_error = str(e)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Siesta is awake at {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nGoodnight.")
