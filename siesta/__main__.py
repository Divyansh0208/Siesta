"""python -m siesta            open the app
   python -m siesta run ...   make a plan with no window (what a schedule runs)
   python -m siesta schedule  print the commands for a weekly plan"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import draft, engine, plan, server, sources, store


def cmd_run(a) -> int:
    parts = [a.brief or ""]
    try:
        if a.git:
            parts.append(sources.brief_from_git(a.git))
        if a.brief_file:
            parts.append(sources.brief_from_file(a.brief_file))
        if a.csv:
            df = sources.from_csv_text(Path(a.csv).read_text(encoding="utf-8"))
            store.save_history(df, {"kind": "csv", "label": "Your CSV"})
        elif a.dev_user:
            df = sources.fetch_dev(a.dev_user)
            store.save_history(df, {"kind": "dev", "label": f"DEV posts, @{a.dev_user}"})
        else:
            loaded = store.load_history()
            if not loaded:
                print("No history yet. Open the app once (python -m siesta), or pass --csv / --dev-user.")
                return 1
            df = loaded[0]
        brief = "\n".join(p for p in parts if p).strip()
        if not brief:
            print("Give a brief: --brief, --git or --brief-file.")
            return 1
        m = engine.fit(df)
        installed = None if a.no_llm else draft.models()
        model = a.model or (draft.default_model(installed) if installed else None)
        p = plan.make_plan(df, m, brief, a.n, model)
    except engine.HistoryError as e:
        print(e)
        return 1
    path = store.save_plan(p)
    print(p["verdict"]["text"])
    for d in p["drafts"]:
        print(f"  {engine.DAYS[d['wd']]} {d['hour']:02d}:00 UTC  [{d['band']}]  {d['title']}")
    print(f"Saved: {path.with_suffix('.csv')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="siesta")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="open the app (default)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-open", action="store_true")
    r = sub.add_parser("run", help="make a plan without the window")
    r.add_argument("--brief", help="what the posts are about")
    r.add_argument("--git", help="folder: use its recent commits as the brief")
    r.add_argument("--brief-file", help="text file to use as the brief")
    r.add_argument("--csv", help="history CSV to use (saved for next time)")
    r.add_argument("--dev-user", help="fetch this DEV username's public posts")
    r.add_argument("--n", type=int, default=5)
    r.add_argument("--model", help="Gemini model name (default: gemini-2.5-flash)")
    r.add_argument("--no-llm", action="store_true", help="templates only")
    c = sub.add_parser("schedule", help="print weekly schedule commands")
    c.add_argument("--day", default="mon")
    c.add_argument("--time", default="02:00")
    c.add_argument("--git", default="/path/to/your/project")
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "schedule":
        t = plan.schedule_text(a.day, a.time, a.git, "git", sys.executable, str(Path.cwd()))
        print(t["mac_linux"], "", t["windows"], "", t["note"], sep="\n")
        return 0
    port, no_open = (a.port, a.no_open) if a.cmd == "serve" else (8765, False)
    server.serve(port, not no_open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
