"""Where things live: one folder, plain files. Set SIESTA_HOME to move it."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from . import engine


def home() -> Path:
    p = Path(os.environ.get("SIESTA_HOME") or Path.cwd() / "data")
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_history(df: pd.DataFrame, meta: dict) -> None:
    out = df.copy()
    out["tags"] = out["tags"].map(",".join)
    out.to_csv(home() / "history.csv", index=False)
    (home() / "history.json").write_text(json.dumps(meta), encoding="utf-8")


def load_history() -> tuple[pd.DataFrame, dict] | None:
    f, m = home() / "history.csv", home() / "history.json"
    if not f.exists():
        return None
    meta = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {"kind": "csv", "label": "Saved history"}
    return engine.load_history(f), meta


def save_plan(plan: dict) -> Path:
    d = home() / "plans"
    d.mkdir(exist_ok=True)
    (d / f"{plan['id']}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    from .plan import plan_csv
    (d / f"{plan['id']}.csv").write_text(plan_csv(plan), encoding="utf-8")
    return d / f"{plan['id']}.json"


def latest_plan() -> dict | None:
    files = sorted((home() / "plans").glob("*.json")) if (home() / "plans").exists() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None
