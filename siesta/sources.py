"""Where history and briefs come from. DEV's public API needs no key; git is optional."""
from __future__ import annotations

import io
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine

DEV_API = "https://dev.to/api/articles"
UA = {"User-Agent": "siesta/0.1 (local planner)", "Accept": "application/json"}
SETTLE_DAYS = 14  # newer posts have not collected their reactions yet


def from_csv_text(text: str) -> pd.DataFrame:
    return engine.load_history(io.StringIO(text))


def _get(url: str, tries: int = 3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise engine.HistoryError(f"DEV answered with an error ({e.code}). Try again in a minute.") from e
        except urllib.error.URLError as e:
            raise engine.HistoryError("Could not reach dev.to. Check your connection, "
                                      "or upload a CSV instead.") from e


def fetch_dev(username: str = "", tag: str = "", days: int = 60, log=print) -> pd.DataFrame:
    """Public posts from DEV: your own (username) or a community tag (cold start).
    Only 6 fields are kept. No author names, no usernames, no post text."""
    username, tag = username.strip().lstrip("@"), tag.strip().lstrip("#").lower()
    if not username and not tag:
        raise engine.HistoryError("Give a DEV username or a tag.")
    key = f"username={username}" if username else f"tag={tag}"
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=SETTLE_DAYS)
    rows, oldest_wanted = [], pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days + SETTLE_DAYS)
    for page in range(1, 40):
        batch = _get(f"{DEV_API}?{key}&per_page=1000&page={page}")
        if not batch:
            break
        for a in batch:
            rows.append({"when": a.get("published_at"), "title": a.get("title") or "",
                         "score": a.get("public_reactions_count"),
                         "tags": ",".join(a.get("tag_list") or []),
                         "minutes": a.get("reading_time_minutes"),
                         "cover": int(bool(a.get("cover_image")))})
        log(f"Fetched {len(rows)} posts")
        last = pd.to_datetime(batch[-1].get("published_at"), utc=True, errors="coerce")
        if len(batch) < 1000 or (tag and pd.notna(last) and last < oldest_wanted):
            break
        time.sleep(1)  # be polite
    df = pd.DataFrame(rows)
    if df.empty:
        raise engine.HistoryError("DEV returned no posts for that name or tag.")
    df["when"] = pd.to_datetime(df["when"], utc=True, errors="coerce")
    df = df[df["when"] <= cutoff]
    if tag:
        df = df[df["when"] >= oldest_wanted]
    df = df.drop_duplicates(subset=["when", "title"])
    return engine.load_history(df)


def demo_history(n: int = 1400, seed: int = 7) -> pd.DataFrame:
    """Made-up posts with a planted, modest pattern. For trying the app offline. Not real data."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=15)
    when = end - pd.to_timedelta(np.sort(rng.uniform(0, 110 * 24, n))[::-1], unit="h")
    topics = ["queues", "caching", "typing", "builds", "tests", "logs", "forms", "routing",
              "backups", "search", "deploys", "configs"]
    frames = ["{} for people in a hurry", "What I learned about {}", "Is {} worth it?",
              "{} without the pain", "{} notes", "Why {} matters"]
    tagpool = ["python", "webdev", "javascript", "beginners", "devops", "ai", "tutorial"]
    rows = []
    for t in when:
        title = frames[rng.integers(len(frames))].format(topics[rng.integers(len(topics))])
        if rng.random() < 0.3:
            title = f"{rng.integers(3, 12)} " + title
        cover, tags = int(rng.random() < 0.3), list(rng.choice(tagpool, rng.integers(1, 5), replace=False))
        lam = 0.35 * np.exp(0.9 * cover + 0.35 * ("?" in title or title[0].isdigit())
                            + 0.25 * (t.dayofweek in (1, 2)) + 0.3 * ("ai" in tags))
        rows.append({"when": t, "title": title, "score": rng.poisson(lam), "tags": ",".join(tags),
                     "minutes": int(rng.integers(2, 12)), "cover": cover})
    return engine.load_history(pd.DataFrame(rows))


def brief_from_git(repo: str, days: int = 14, limit: int = 30) -> str:
    """Recent commit subjects as a brief. Optional; needs the git command."""
    if not Path(repo).expanduser().is_dir():
        raise engine.HistoryError(f"Not a folder: {repo}")
    try:
        out = subprocess.run(["git", "-C", str(Path(repo).expanduser()), "log", f"--since={days}.days",
                              "--no-merges", "--pretty=%s", f"-n{limit}"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        raise engine.HistoryError("Could not read git history from that folder.") from e
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return ("What I worked on recently: " + "; ".join(lines)) if lines else ""


def brief_from_file(path: str, limit: int = 1500) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        raise engine.HistoryError(f"Not a file: {path}")
    return p.read_text(encoding="utf-8", errors="replace")[:limit].strip()
