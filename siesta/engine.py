"""History in, honest answer out.

Load any post history, turn each post into numbers known before publishing, train a small
logistic regression on the older posts, and test it on the newest ones. If it cannot beat
guessing, say so. Everything here is plain numpy / pandas / scikit-learn: no keys, no downloads.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 60
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BLOCKS = [f"{h:02d}:00-{h + 3:02d}:59" for h in range(0, 24, 4)]
ALIASES = {
    "when": ["when", "published_at", "post_datetime", "created_at", "date", "datetime",
             "timestamp", "time", "posted_at"],
    "title": ["title", "headline", "caption", "text", "name"],
    "score": ["score", "public_reactions_count", "reactions", "likes", "engagement", "views",
              "claps"],
    "tags": ["tags", "tag_list", "hashtags"],
    "minutes": ["minutes", "reading_time_minutes", "read_time", "reading_time"],
    "cover": ["cover", "has_cover_image", "cover_image", "has_image", "image"],
}
NUMERIC = {"title_len": "title length", "n_tags": "number of tags", "minutes": "reading time"}
YES = {"cover": "a cover image", "has_question": "a question in the title",
       "has_number": "a number in the title"}
NO = {"cover": "no cover image", "has_question": "no question in the title",
      "has_number": "no number in the title"}
MIN_EFFECT = 0.01  # a change of under one point in P(hit) is not worth suggesting


class HistoryError(ValueError):
    """The history can't be used. The message says why, in plain words."""


# ---------- loading ----------

def parse_tags(v: object) -> list[str]:
    if isinstance(v, (list, tuple)):
        items = [str(t) for t in v]
    elif v is None or (isinstance(v, float) and math.isnan(v)):
        items = []
    else:
        items = re.split(r"[,\s]+", re.sub(r"[\[\]'\"#]", " ", str(v)))
    return [t.strip().lower() for t in items if t.strip()]


def _flag(v: object) -> int:
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
        return int(v != 0)
    return int(str(v).strip().lower() not in ("", "0", "false", "nan", "none", "null"))


def load_history(src) -> pd.DataFrame:
    """Path, file object or DataFrame -> clean frame: when, title, score, tags, minutes, cover."""
    raw = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    cols = {str(c).lower().strip(): c for c in raw.columns}
    pick = {k: next((cols[a] for a in al if a in cols), None) for k, al in ALIASES.items()}
    missing = [k for k in ("when", "title", "score") if pick[k] is None]
    if missing:
        raise HistoryError("I could not find a column for: " + ", ".join(missing)
                           + ". Columns in the file: " + ", ".join(map(str, raw.columns[:12])))
    n = len(raw)
    df = pd.DataFrame({
        "when": pd.to_datetime(raw[pick["when"]], errors="coerce", utc=True),
        "title": raw[pick["title"]].fillna("").astype(str),
        "score": pd.to_numeric(raw[pick["score"]], errors="coerce"),
    })
    df["tags"] = (raw[pick["tags"]].map(parse_tags) if pick["tags"]
                  else pd.Series([[] for _ in range(n)], index=raw.index, dtype=object))
    mins = pd.to_numeric(raw[pick["minutes"]], errors="coerce") if pick["minutes"] else None
    df["minutes"] = mins if mins is not None else np.nan
    df["minutes"] = df["minutes"].fillna(df["minutes"].median() if df["minutes"].notna().any() else 5)
    df["cover"] = raw[pick["cover"]].map(_flag) if pick["cover"] else 0
    df = df.dropna(subset=["when", "score"]).sort_values("when", kind="stable").reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise HistoryError(f"Only {len(df)} usable posts. I need at least {MIN_ROWS} to learn "
                           "anything; more is better.")
    return df


# ---------- features ----------

def top_tags(df: pd.DataFrame, k: int = 8, min_n: int = 10) -> list[str]:
    counts = pd.Series([t for ts in df["tags"] for t in ts]).value_counts()
    return [str(t) for t in counts[counts >= min_n].index[:k]]


def featurize(df: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """Only things known before posting. Same code for real posts and drafts."""
    t = df["title"].fillna("")
    f = pd.DataFrame(index=df.index)
    f["title_len"] = t.str.len().clip(upper=150)
    f["has_question"] = t.str.contains("?", regex=False).astype(int)
    f["has_number"] = t.str.contains(r"\d", regex=True).astype(int)
    f["n_tags"] = df["tags"].map(len)
    f["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(5).clip(upper=30)
    f["cover"] = df["cover"].astype(int)
    wd, hb = df["when"].dt.dayofweek, df["when"].dt.hour // 4
    for i in range(7):
        f[f"wd{i}"] = (wd == i).astype(int)
    for i in range(6):
        f[f"hb{i}"] = (hb == i).astype(int)
    for tg in tags:
        f[f"tag:{tg}"] = df["tags"].map(lambda x, tg=tg: int(tg in x))
    return f


def draft_row(d: dict) -> pd.DataFrame:
    """One draft -> a one-row frame that featurize() understands."""
    when = pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(days=int(d["wd"]), hours=int(d["hour"]))
    return pd.DataFrame([{"when": when, "title": d["title"], "tags": parse_tags(d["tags"]),
                          "minutes": d["minutes"], "cover": int(bool(d["cover"]))}])


# ---------- backtest ----------

def hit_threshold(train_scores: pd.Series) -> float:
    """A 'hit' is at least the median training score, and never less than 1."""
    return max(1.0, float(np.median(train_scores)))


def _stats(y: np.ndarray, p: np.ndarray):
    if y.min() == y.max():
        return None
    k = max(1, math.ceil(0.2 * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    return float(roc_auc_score(y, p)), float(y[top].mean() / y.mean())


def evaluate(y: np.ndarray, p: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict:
    full = _stats(y, p)
    out = {"n_test": int(len(y)), "hit_rate": float(y.mean())}
    if full is None:
        return {**out, "auc": None, "auc_ci": None, "lift": None, "lift_ci": None,
                "kind": "unknown",
                "verdict": "The newest posts all have the same outcome, so I can't test anything."}
    rng = np.random.default_rng(seed)
    boots = [s for s in (_stats(y[i], p[i]) for i in
                         (rng.integers(0, len(y), len(y)) for _ in range(n_boot))) if s]
    a = np.percentile([b[0] for b in boots], [2.5, 97.5])
    l = np.percentile([b[1] for b in boots], [2.5, 97.5])
    out.update(auc=full[0], auc_ci=[float(a[0]), float(a[1])], lift=full[1],
               lift_ci=[float(l[0]), float(l[1])])
    if out["auc_ci"][0] > 0.5:
        out["kind"] = "signal"
        out["verdict"] = (f"A weak but real pattern. It ranks newer posts better than chance "
                          f"(AUC {full[0]:.2f}, range {a[0]:.2f} to {a[1]:.2f}). Use it to order "
                          "ideas, not to promise results.")
    else:
        out["kind"] = "noise"
        out["verdict"] = (f"No pattern worth chasing. It does not separate hits from misses better "
                          f"than guessing (AUC {full[0]:.2f}, range {a[0]:.2f} to {a[1]:.2f}, "
                          "and 0.50 is a coin flip). Plans stay unranked.")
    if len(y) < 100:
        out["verdict"] += " Small test window: treat this with extra care."
    return out


@dataclass
class Model:
    pipe: object
    cols: list[str]
    tags: list[str]
    thr: float
    cuts: tuple[float, float]  # (strong at or above, weak below)
    report: dict
    heat: list = field(default_factory=list)

    @property
    def signal(self) -> bool:
        return self.report["kind"] == "signal"


def fit(df: pd.DataFrame, seed: int = 0) -> Model:
    """Train on the older 80%, test on the newest 20% (at least 30 posts). Nothing from the
    test window reaches the model, the hit threshold, or the tag list."""
    n = len(df)
    n_test = max(30, round(n * 0.2))
    train, test = df.iloc[: n - n_test], df.iloc[n - n_test:]
    if len(train) < 30:
        raise HistoryError("Not enough older posts to train on. Add more history.")
    thr, tags = hit_threshold(train["score"]), top_tags(train)
    xtr, xte = featurize(train, tags), featurize(test, tags)
    ytr, yte = (train["score"] >= thr).to_numpy(int), (test["score"] >= thr).to_numpy(int)
    if ytr.min() == ytr.max():
        raise HistoryError("Every older post has the same outcome, so there is nothing to learn "
                           "from. Try a longer history.")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(xtr, ytr)
    p = pipe.predict_proba(xte)[:, 1]
    rep = evaluate(yte, p, seed=seed)
    gb = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=seed).fit(xtr, ytr)
    rep["flex_auc"] = _stats(yte, gb.predict_proba(xte)[:, 1])
    rep["flex_auc"] = None if rep["flex_auc"] is None else rep["flex_auc"][0]
    strong, weak = float(np.quantile(p, 0.8)), float(np.quantile(p, 0.4))
    bands = {}
    for name, mask in (("Strong", p >= strong), ("Toss-up", (p < strong) & (p >= weak)),
                       ("Weak", p < weak)):
        bands[name] = {"n": int(mask.sum()), "hit": float(yte[mask].mean()) if mask.any() else None}
    rep.update(bands=bands, thr=thr, n_train=int(len(train)),
               train_hit=float(ytr.mean()), tags=tags,
               range=[str(df["when"].iloc[0].date()), str(df["when"].iloc[-1].date())])
    hit_all = (df["score"] >= hit_threshold(df["score"])).astype(int)
    g = pd.DataFrame({"wd": df["when"].dt.dayofweek, "hb": df["when"].dt.hour // 4, "hit": hit_all})
    cell = g.groupby(["wd", "hb"])["hit"].agg(["size", "mean"])
    heat = [[{"n": int(cell.loc[(w, b), "size"]) if (w, b) in cell.index else 0,
              "rate": float(cell.loc[(w, b), "mean"]) if (w, b) in cell.index else None}
             for b in range(6)] for w in range(7)]
    return Model(pipe, list(xtr.columns), tags, thr, (strong, weak), rep, heat)


# ---------- scoring drafts ----------

def _features(m: Model, d: dict) -> pd.DataFrame:
    return featurize(draft_row(d), m.tags)[m.cols]


def band_for(m: Model, p: float) -> str:
    if not m.signal:
        return "Unranked"
    return "Strong" if p >= m.cuts[0] else "Weak" if p < m.cuts[1] else "Toss-up"


def _with_slot(x: pd.Series, wd: int, hb: int) -> pd.Series:
    v = x.copy()
    for i in range(7):
        v[f"wd{i}"] = int(i == wd)
    for i in range(6):
        v[f"hb{i}"] = int(i == hb)
    return v


def best_slots(m: Model, d: dict) -> list[tuple[float, int, int]]:
    """All 42 (weekday, 4-hour window) slots for this draft, best first, as (p, wd, hb)."""
    x = _features(m, d).iloc[0]
    slots = [(w, b) for w in range(7) for b in range(6)]
    p = m.pipe.predict_proba(pd.DataFrame([_with_slot(x, w, b) for w, b in slots])[m.cols])[:, 1]
    return sorted(((float(p[i]), *slots[i]) for i in range(len(slots))), reverse=True)


def explain(m: Model, x: pd.DataFrame) -> list[dict]:
    sc, lr = m.pipe[0], m.pipe[1]
    v = x.iloc[0].to_numpy(float)
    z = (v - sc.mean_) / sc.scale_
    out = []
    for i in np.argsort(-np.abs(lr.coef_[0] * z)):
        c, val, w = m.cols[i], v[i], float(lr.coef_[0][i] * z[i])
        if abs(w) < 0.02:
            break
        if c in NUMERIC:
            text = f"{'longer' if z[i] > 0 else 'shorter'} {NUMERIC[c]}"
        elif c in YES:
            text = YES[c] if val else NO[c]
        elif not val:
            continue  # one-hot that is off says nothing useful
        elif c.startswith("wd"):
            text = f"posting on {DAYS[int(c[2:])]}"
        elif c.startswith("hb"):
            text = f"posting {BLOCKS[int(c[2:])]} UTC"
        else:
            text = f"the tag '{c[4:]}'"
        out.append({"text": text, "up": w > 0})
        if len(out) == 3:
            break
    return out


def ideas(m: Model, d: dict, p0: float) -> list[dict]:
    """One-change what-ifs, best per kind. They are the model's guess, not proof."""
    x = _features(m, d).iloc[0]
    groups: dict[str, list[tuple[str, pd.Series]]] = {}

    def add(kind, label, **ov):
        v = x.copy()
        for k, val in ov.items():
            v[k] = val
        groups.setdefault(kind, []).append((label, v))

    add("cover", "Add a cover image" if not x["cover"] else "Drop the cover image",
        cover=1 - x["cover"])
    if not x["has_question"]:
        add("q", "Ask a question in the title", has_question=1)
    if not x["has_number"]:
        add("n", "Put a number in the title", has_number=1)
    cur = (int(np.argmax([x[f"wd{i}"] for i in range(7)])), int(np.argmax([x[f"hb{i}"] for i in range(6)])))
    for w in range(7):
        for b in range(6):
            if (w, b) != cur:
                v = _with_slot(x, w, b)
                groups.setdefault("slot", []).append((f"Post on {DAYS[w]} {BLOCKS[b]} UTC", v))
    for tg in m.tags:
        if f"tag:{tg}" in x and not x[f"tag:{tg}"]:
            add("tag", f"Add the tag '{tg}'", **{f"tag:{tg}": 1, "n_tags": x["n_tags"] + 1})
    out = []
    for opts in groups.values():
        p = m.pipe.predict_proba(pd.DataFrame([v for _, v in opts])[m.cols])[:, 1]
        i = int(np.argmax(p))
        if p[i] - p0 >= MIN_EFFECT:
            out.append({"text": opts[i][0], "p": float(p[i]), "delta": float(p[i] - p0)})
    return sorted(out, key=lambda o: -o["delta"])[:3]


def score_draft(m: Model, d: dict) -> dict:
    x = _features(m, d)
    p = float(m.pipe.predict_proba(x)[0, 1])
    return {"p": p, "band": band_for(m, p), "why": explain(m, x), "ideas": ideas(m, d, p)}
