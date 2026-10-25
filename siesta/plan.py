"""Tie it together: history + brief -> a week of scored drafts."""
from __future__ import annotations

import csv
import io
import re
from datetime import datetime

import pandas as pd

from . import draft, engine

SPREAD = [(1, 14), (2, 14), (3, 14), (0, 14), (4, 14), (5, 14), (6, 14)]  # used when timing carries no signal


def verdict(drafts: list[dict], m: engine.Model) -> dict:
    n = sum(d["band"] == "Strong" for d in drafts)
    if not drafts:
        text = "I could not draft anything. See the log."
    elif not m.signal:
        text = "The history shows nothing to rank by, so these are unranked. Pick what you like, then go back to sleep."
    elif n == 0:
        text = "Nothing stands out this week. Go back to sleep."
    else:
        text = f"{n} strong idea{'s' if n != 1 else ''} this week."
    return {"text": text, "n_strong": n, "kind": "sleep" if n == 0 else "go"}


def make_plan(df: pd.DataFrame, m: engine.Model, brief: str, n: int = 5, model: str | None = None,
              ask_fn=draft.chat, log=print) -> dict:
    allowed = engine.top_tags(df, k=60, min_n=3)
    typical = float(df["minutes"].median())
    mode, drafts = "templates", None
    if model:
        try:
            drafts, mode = draft.generate(model, brief, n, allowed, ask_fn, log), model
        except draft.LLMUnavailable as e:
            log(f"{e} Using plain templates instead.")
    if drafts is None:
        drafts = draft.template_drafts(brief, n, allowed, typical)
    log("Scoring against your history")
    used: set[int] = set()
    for i, d in enumerate(drafts):
        d["cover"] = 0
        d["wd"], d["hour"] = (2, 14)
        if m.signal:
            for _p, wd, hb in engine.best_slots(m, d):
                if wd not in used or len(used) >= 7:
                    d["wd"], d["hour"] = wd, hb * 4
                    used.add(wd)
                    break
        else:
            d["wd"], d["hour"] = SPREAD[i % 7]
        d.update(engine.score_draft(m, d))
    if m.signal and mode != "templates":
        todo = [d for d in drafts if d["band"] != "Strong"]
        for k, d in enumerate(todo, 1):
            log(f"Polishing {k} of {len(todo)}")
            hints = [i["text"] for i in d["ideas"] if not i["text"].startswith("Post on")]
            cur = {x: d[x] for x in ("title", "tags", "outline", "minutes")}
            try:
                new, _ = draft.ask(mode, brief, "same angle", allowed, ask_fn, current=cur, hints=hints)
            except draft.LLMUnavailable:
                break
            if not new:
                continue
            cand = {**d, **new}
            s = engine.score_draft(m, cand)
            if s["p"] > d["p"]:  # a revision has to earn its place
                cand.update(s)
                cand["original"] = {"title": d["title"], "p": d["p"], "band": d["band"]}
                d.clear()
                d.update(cand)
    drafts.sort(key=lambda d: (d["wd"], d["hour"]))
    now = datetime.now()
    r = m.report
    return {"id": now.strftime("%Y%m%d-%H%M%S"), "created": now.isoformat(timespec="seconds"),
            "brief": " ".join(brief.split())[:800], "mode": mode, "verdict": verdict(drafts, m),
            "signal": m.signal, "auc": r["auc"], "auc_ci": r["auc_ci"], "drafts": drafts}


def plan_csv(plan: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["verdict", "day_utc", "hour_utc", "title", "tags", "minutes", "band", "score", "outline"])
    for d in plan["drafts"]:
        w.writerow([plan["verdict"]["text"], engine.DAYS[d["wd"]], d["hour"], d["title"],
                    ",".join(d["tags"]), d["minutes"], d["band"], f"{d['p']:.3f}",
                    " | ".join(d["outline"])])
    return buf.getvalue()


def schedule_text(day: str, time: str, source: str, kind: str, python: str, root: str) -> dict:
    """The commands that would run a plan weekly. Nothing is created for you."""
    days = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
    day = day.lower()[:3]
    if day not in days or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", time):
        raise ValueError("Pick a weekday and a time like 02:00.")
    if any(c in source for c in '"\n\r\0'):
        raise ValueError("The path cannot contain quotes or line breaks.")
    flag = {"git": "--git", "file": "--brief-file"}.get(kind)
    if not flag:
        raise ValueError("Brief source must be 'git' or 'file'.")
    h, mi = time.split(":")
    run = f'-m siesta run {flag} "{source}"'
    return {
        "mac_linux": f'{int(mi)} {int(h)} * * {days.index(day)} cd "{root}" && "{python}" {run} '
                     '>> data/siesta.log 2>&1   # add with: crontab -e',
        "windows": f'schtasks /Create /SC WEEKLY /D {day.upper()} /TN Siesta /ST {time} /F /TR '
                   f'"cmd /c cd /d \\"{root}\\" && \\"{python}\\" {run.replace(chr(34), chr(92) + chr(34))}"',
        "note": "Your computer must be on and awake at that time, and GEMINI_API_KEY set if you want "
                "model-written drafts. Nothing has been scheduled; copy a line if you want it.",
    }
