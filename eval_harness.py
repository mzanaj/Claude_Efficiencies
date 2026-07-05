#!/usr/bin/env python3
"""
eval_harness.py — QA harness for LLM-produced training labels.

Design (blind re-annotation, not verification):
  1. Stratified sample from your labeled CSV: a FIXED CORE that never changes
     (regression set) + a FRESH rotation per class (coverage/drift).
  2. Pass 1 — a judge model RE-LABELS each example BLIND (it never sees your
     label), one example per call, temperature 0, cached.
  3. Harness computes agreement, Cohen's kappa, per-class agreement,
     labeler-vs-judge confusion matrix.
  4. Pass 2 — the judge critiques ONLY the disagreements: crux, issue tag,
     recommended label, rubric gaps.
  5. Outputs: results_<run>.csv (everything), review_queue_<run>.csv (for the
     human), runs_log.csv (one row per run — this is where drift lives).

The judge reads judge_rubric_v1.md — an independent statement of the task —
NOT the labeler's prompt. Use a different model family than your labeler when
possible.

CSV schema (input): columns `text`, `llm_label` with values in
{positive, negative, unsure}. Optional `id` column (else sha1(text) is used).

Optional golden_set.csv (columns: text or id, human_label): if present, the
harness also reports labeler accuracy AND judge accuracy against human labels
— i.e. it calibrates the judge itself.

Usage:
  pip install anthropic pandas
  export ANTHROPIC_API_KEY=sk-ant-...
  python eval_harness.py --csv data.csv --run-id v7 --labeler-version prompt_v7
  python eval_harness.py --ingest-queue eval_out/review_queue_v7.csv  # adjudications -> golden set + core
  python eval_harness.py --compare-last-two          # drift report only

Prompt-change workflow: relabel your data with the new labeler prompt, rerun
with a new --run-id/--labeler-version, then --compare-last-two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
JUDGE_MODEL = "claude-sonnet-4-6"  # prefer a different family than your labeler
JUDGE_PROMPT_VERSION = "judge_v1"
CLASSES = ["positive", "negative", "unsure"]
CORE_PER_CLASS = 15                # size of the bootstrapped fixed regression core

OUT_DIR = Path("eval_out")
CACHE_PATH = OUT_DIR / "judge_cache.jsonl"
RUNS_LOG = OUT_DIR / "runs_log.csv"
FIXED_CORE_PATH = OUT_DIR / "fixed_core_ids.txt"
GOLDEN_PATH = Path("golden_set.csv")

ISSUE_TAGS = [
    "keyword_trap",       # surface words resemble the other class
    "borderline_intent",  # intent genuinely sits between classes
    "missing_context",    # text alone cannot support a confident label
    "rubric_gap",         # the rubric is silent/ambiguous on this case
    "multi_topic",        # multiple topics, label depends on which span you weigh
    "annotator_error",    # one side plainly misapplied a clear rule
]

# --------------------------------------------------------------------------
# Judge prompts (<<TOKENS>> are replaced at runtime; never f-strings)
# --------------------------------------------------------------------------
PASS1_SYSTEM = """You are an independent annotator producing a second, blind label for quality control of classifier training data. You have not seen any other annotator's label and must not try to guess it. Apply the rubric below exactly as written, not your own domain intuition.

<rubric>
<<RUBRIC>>
</rubric>

Rules:
- Choose exactly one label: positive, negative, or unsure.
- "unsure" is a real label for text the rubric's decision rules cannot resolve. If you choose it, name the rule that failed in your rationale.
- Judge meaning and intent, not surface keywords.

Respond with ONLY a JSON object, no markdown fences:
{"label": "positive|negative|unsure", "confidence": "high|medium|low", "rationale": "<max 25 words>"}"""

PASS1_USER = """TEXT TO LABEL:
<<TEXT>>"""

PASS2_SYSTEM = """You are a QA reviewer resolving a labeling disagreement between two annotators who used the same rubric: annotator A (the production labeler) and annotator B (an independent blind annotator).

<rubric>
<<RUBRIC>>
</rubric>

Decide which label the rubric actually supports, state the crux of the disagreement in one sentence, and classify the issue with exactly one tag from:
<<TAGS>>
Use "other:<short-slug>" only if none fit. If the rubric itself is silent or ambiguous on this case, describe the gap in one sentence in "rubric_gap"; otherwise set it to null.

Respond with ONLY a JSON object, no markdown fences:
{"recommended_label": "positive|negative|unsure", "crux": "<one sentence>", "issue_tag": "<tag>", "rubric_gap": "<one sentence or null>"}"""

PASS2_USER = """TEXT:
<<TEXT>>

Label A (production labeler): <<LABEL_A>>
Label B (blind annotator): <<LABEL_B>>"""


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def sha_id(text: str) -> str:
    return hashlib.sha1(str(text).strip().encode("utf-8")).hexdigest()[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Data loading + sampling
# --------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"text", "llm_label"} - set(df.columns)
    if missing:
        sys.exit(f"input CSV is missing columns: {sorted(missing)}")
    df["llm_label"] = df["llm_label"].astype(str).str.strip().str.lower()
    bad = df[~df["llm_label"].isin(CLASSES)]
    if len(bad):
        print(f"[load] dropping {len(bad)} rows with labels outside {CLASSES}")
        df = df[df["llm_label"].isin(CLASSES)]
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = df["text"].map(sha_id)
    df = df.drop_duplicates("id").reset_index(drop=True)
    return df


def load_fixed_core() -> set[str]:
    if FIXED_CORE_PATH.exists():
        return {l.strip() for l in FIXED_CORE_PATH.read_text().splitlines() if l.strip()}
    return set()


def make_slice(df: pd.DataFrame, n_fresh: int, seed: int) -> pd.DataFrame:
    """Fixed regression core (same every run) + fresh stratified rotation."""
    core_ids = load_fixed_core()
    core = df[df["id"].isin(core_ids)]
    pool = df[~df["id"].isin(core_ids)]

    fresh_parts = []
    for c in CLASSES:
        cand = pool[pool["llm_label"] == c]
        n = min(n_fresh, len(cand))
        if n:
            fresh_parts.append(cand.sample(n=n, random_state=seed))
    fresh = pd.concat(fresh_parts) if fresh_parts else pool.iloc[0:0]

    if not core_ids:  # first run: freeze part of this sample as the regression core
        boot = []
        for c in CLASSES:
            boot.extend(fresh[fresh["llm_label"] == c]["id"].head(CORE_PER_CLASS).tolist())
        FIXED_CORE_PATH.write_text("\n".join(sorted(boot)) + "\n")
        print(f"[core] bootstrapped fixed regression core: {len(boot)} ids -> {FIXED_CORE_PATH}")

    out = pd.concat([core, fresh]).drop_duplicates("id").reset_index(drop=True)
    counts = out["llm_label"].value_counts().to_dict()
    print(f"[slice] {len(out)} examples ({counts}; {len(core)} from fixed core)")
    return out


# --------------------------------------------------------------------------
# Cache (never pay twice for the same judgment)
# --------------------------------------------------------------------------
def load_cache() -> dict:
    cache = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec["value"]
    return cache


def cache_put(key: str, value: dict) -> None:
    with CACHE_PATH.open("a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


# --------------------------------------------------------------------------
# Judge calls
# --------------------------------------------------------------------------
def get_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("missing dependency: pip install anthropic (and set ANTHROPIC_API_KEY)")
    return anthropic.Anthropic()


def call_llm(client, system: str, user: str, max_tokens: int = 300) -> str:
    delay = 2.0
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except Exception:
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def parse_json_block(s: str):
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def judge_pass1(df: pd.DataFrame, rubric: str, use_cache: bool = True) -> pd.DataFrame:
    """Blind re-annotation: the judge labels each example without seeing ours."""
    cache = load_cache() if use_cache else {}
    client = None
    rub_h = hashlib.sha1(rubric.encode()).hexdigest()[:8]
    system = PASS1_SYSTEM.replace("<<RUBRIC>>", rubric)

    rows = []
    for i, r in df.iterrows():
        key = f"p1|{JUDGE_MODEL}|{JUDGE_PROMPT_VERSION}|{rub_h}|{r['id']}"
        if key in cache:
            v = cache[key]
        else:
            if client is None:
                client = get_client()
            raw = call_llm(client, system, PASS1_USER.replace("<<TEXT>>", str(r["text"])))
            v = parse_json_block(raw) or {}
            v["_raw"] = raw
            if use_cache:
                cache_put(key, v)
            cache[key] = v
        lab = str(v.get("label", "")).strip().lower()
        rows.append({
            "id": r["id"],
            "text": r["text"],
            "llm_label": r["llm_label"],
            "judge_label": lab if lab in CLASSES else None,
            "judge_confidence": v.get("confidence"),
            "judge_rationale": v.get("rationale"),
            "parse_error": lab not in CLASSES,
        })
        print(f"[pass1] {i + 1}/{len(df)}", end="\r")
    print()
    return pd.DataFrame(rows)


def judge_pass2(res: pd.DataFrame, rubric: str, use_cache: bool = True) -> pd.DataFrame:
    """Critique pass, run only on disagreements: crux + issue tag + recommendation."""
    for col in ("recommended_label", "crux", "issue_tag", "rubric_gap"):
        if col not in res.columns:
            res[col] = None

    dis = res[res["judge_label"].notna() & (res["judge_label"] != res["llm_label"])]
    if dis.empty:
        print("[pass2] no disagreements — skipping")
        return res

    cache = load_cache() if use_cache else {}
    client = None
    rub_h = hashlib.sha1(rubric.encode()).hexdigest()[:8]
    system = (PASS2_SYSTEM
              .replace("<<RUBRIC>>", rubric)
              .replace("<<TAGS>>", ", ".join(ISSUE_TAGS)))

    done = 0
    for idx, r in dis.iterrows():
        key = f"p2|{JUDGE_MODEL}|{JUDGE_PROMPT_VERSION}|{rub_h}|{r['id']}|{r['llm_label']}|{r['judge_label']}"
        if key in cache:
            v = cache[key]
        else:
            if client is None:
                client = get_client()
            user = (PASS2_USER
                    .replace("<<TEXT>>", str(r["text"]))
                    .replace("<<LABEL_A>>", str(r["llm_label"]))
                    .replace("<<LABEL_B>>", str(r["judge_label"])))
            raw = call_llm(client, system, user)
            v = parse_json_block(raw) or {}
            v["_raw"] = raw
            if use_cache:
                cache_put(key, v)
            cache[key] = v
        res.loc[idx, "recommended_label"] = v.get("recommended_label")
        res.loc[idx, "crux"] = v.get("crux")
        res.loc[idx, "issue_tag"] = v.get("issue_tag")
        res.loc[idx, "rubric_gap"] = v.get("rubric_gap")
        done += 1
        print(f"[pass2] {done}/{len(dis)}", end="\r")
    print()
    return res


# --------------------------------------------------------------------------
# Metrics (computed by code, never by the judge)
# --------------------------------------------------------------------------
def cohen_kappa(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return float("nan")
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    labels = {x for x, _ in pairs} | {y for _, y in pairs}
    pe = sum(
        (sum(1 for x, _ in pairs if x == l) / n) * (sum(1 for _, y in pairs if y == l) / n)
        for l in labels
    )
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def label_dist(series: pd.Series) -> dict:
    vc = series.value_counts(normalize=True)
    return {c: round(float(vc.get(c, 0.0)), 4) for c in CLASSES}


def psi(p: dict, q: dict, eps: float = 1e-4) -> float:
    """Population stability index between two class distributions.
    Rule of thumb: <0.10 stable, 0.10-0.25 moderate shift, >0.25 major shift."""
    total = 0.0
    for c in CLASSES:
        pi = max(p.get(c, 0.0), eps)
        qi = max(q.get(c, 0.0), eps)
        total += (pi - qi) * math.log(pi / qi)
    return total


def per_class_agreement(valid: pd.DataFrame) -> dict:
    out = {}
    for c in CLASSES:
        sub = valid[valid["llm_label"] == c]
        out[c] = round(float((sub["judge_label"] == c).mean()), 4) if len(sub) else float("nan")
    return out


def top_issue_tags(res: pd.DataFrame, k: int = 5) -> dict:
    if "issue_tag" not in res.columns:
        return {}
    vc = res["issue_tag"].dropna().value_counts()
    return {str(t): int(n) for t, n in vc.head(k).items()}


def golden_check(res: pd.DataFrame):
    """If a human-labeled golden set exists, score BOTH the labeler and the judge
    against it. This is how you calibrate the judge itself."""
    if not GOLDEN_PATH.exists():
        return None
    g = pd.read_csv(GOLDEN_PATH)
    if "human_label" not in g.columns:
        return None
    if "id" not in g.columns:
        g = g.copy()
        g["id"] = g["text"].map(sha_id)
    g["human_label"] = g["human_label"].astype(str).str.strip().str.lower()
    m = res.merge(g[["id", "human_label"]], on="id", how="inner")
    if m.empty:
        return None
    mv = m[m["judge_label"].notna()]
    return {
        "n": int(len(m)),
        "labeler_acc": round(float((m["llm_label"] == m["human_label"]).mean()), 4),
        "judge_acc": round(float((mv["judge_label"] == mv["human_label"]).mean()), 4) if len(mv) else float("nan"),
    }


# --------------------------------------------------------------------------
# Runs log + drift
# --------------------------------------------------------------------------
def append_run_log(row: dict) -> None:
    df = pd.DataFrame([row])
    if RUNS_LOG.exists():
        df = pd.concat([pd.read_csv(RUNS_LOG), df], ignore_index=True)
    df.to_csv(RUNS_LOG, index=False)


def _psi_verdict(v: float) -> str:
    if v < 0.10:
        return "stable"
    if v < 0.25:
        return "moderate shift"
    return "MAJOR shift"


def compare_last_two() -> None:
    if not RUNS_LOG.exists():
        sys.exit("no runs_log.csv yet — do at least two runs first")
    log = pd.read_csv(RUNS_LOG)
    if len(log) < 2:
        sys.exit("need at least two runs to compare")
    a, b = log.iloc[-2], log.iloc[-1]
    print(f"\n=== drift: {a['run_id']} -> {b['run_id']} ===")
    for col in ["agreement", "kappa", "agr_positive", "agr_negative", "agr_unsure"]:
        try:
            d = float(b[col]) - float(a[col])
            print(f"  {col:14s} {float(a[col]):.3f} -> {float(b[col]):.3f}  (delta {d:+.3f})")
        except (ValueError, TypeError):
            pass
    for name, desc in [("dist_labeler", "labeler label distribution"),
                       ("dist_judge", "judge label distribution")]:
        try:
            v = psi(json.loads(a[name]), json.loads(b[name]))
            print(f"  PSI {desc}: {v:.4f} ({_psi_verdict(v)})")
        except (ValueError, TypeError, KeyError):
            pass
    print()


# --------------------------------------------------------------------------
# Review queue for the human
# --------------------------------------------------------------------------
def write_review_queue(res: pd.DataFrame, run_id: str) -> Path:
    for col in ("recommended_label", "crux", "issue_tag", "rubric_gap"):
        if col not in res.columns:
            res[col] = None
    q = res[
        res["judge_label"].isna()
        | (res["judge_label"] != res["llm_label"])
        | (res["judge_confidence"] == "low")
    ].copy()
    q["disagree"] = q["judge_label"].notna() & (q["judge_label"] != q["llm_label"])
    conf_rank = {"high": 0, "medium": 1, "low": 2}
    q["_conf"] = q["judge_confidence"].map(conf_rank).fillna(3)
    q = q.sort_values(["disagree", "_conf"], ascending=[False, True])
    q["human_label"] = ""  # fill this in; append adjudicated rows to golden_set.csv
    cols = ["id", "llm_label", "judge_label", "judge_confidence", "recommended_label",
            "issue_tag", "crux", "rubric_gap", "judge_rationale", "text", "human_label"]
    path = OUT_DIR / f"review_queue_{run_id}.csv"
    q[cols].to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Ingest human adjudications
# --------------------------------------------------------------------------
def ingest_queue(queue_path: str) -> None:
    """Read a review queue whose human_label column has been filled in, append
    those rows to golden_set.csv (dedup by id, latest decision wins), and add
    their ids to the fixed core so they recur — and get re-checked — every run."""
    ensure_dirs()
    qp = Path(queue_path)
    if not qp.exists():
        sys.exit(f"queue file not found: {qp}")
    q = pd.read_csv(qp)
    if "human_label" not in q.columns:
        sys.exit("queue file has no human_label column")
    q["human_label"] = q["human_label"].astype(str).str.strip().str.lower()
    adj = q[q["human_label"].isin(CLASSES)][["id", "text", "human_label"]]
    if adj.empty:
        sys.exit(f"no rows with a filled human_label (must be one of: {', '.join(CLASSES)})")

    if GOLDEN_PATH.exists():
        g = pd.read_csv(GOLDEN_PATH)
        if "id" not in g.columns:
            g = g.copy()
            g["id"] = g["text"].map(sha_id)
        combined = pd.concat([g, adj], ignore_index=True)
    else:
        combined = adj
    combined = combined.drop_duplicates("id", keep="last")
    combined.to_csv(GOLDEN_PATH, index=False)

    core = load_fixed_core() | set(adj["id"])
    FIXED_CORE_PATH.write_text("\n".join(sorted(core)) + "\n")
    print(f"[ingest] {len(adj)} adjudications -> {GOLDEN_PATH} (golden set now {len(combined)} rows)")
    print(f"[ingest] ids added to fixed core -> {FIXED_CORE_PATH} ({len(core)} ids total)")


# --------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------
def run(args) -> None:
    ensure_dirs()
    rubric_path = Path(args.rubric)
    if not rubric_path.exists():
        sys.exit(f"rubric file not found: {rubric_path}")
    rubric = rubric_path.read_text()

    df = load_data(args.csv)
    sl = make_slice(df, args.n_fresh, args.seed)

    res = judge_pass1(sl, rubric, use_cache=not args.no_cache)
    if not args.skip_pass2:
        res = judge_pass2(res, rubric, use_cache=not args.no_cache)

    valid = res[res["judge_label"].notna()]
    n_err = int(res["parse_error"].sum())
    agreement = round(float((valid["judge_label"] == valid["llm_label"]).mean()), 4) if len(valid) else float("nan")
    kappa = round(cohen_kappa(valid["llm_label"].tolist(), valid["judge_label"].tolist()), 4)
    pca = per_class_agreement(valid)
    issues = top_issue_tags(res)

    print(f"\n=== run {args.run_id} | labeler={args.labeler_version} | judge={JUDGE_MODEL}/{JUDGE_PROMPT_VERSION} ===")
    print(f"n={len(res)} (parse errors: {n_err})")
    print(f"agreement={agreement}  kappa={kappa}")
    print(f"per-class agreement: {pca}")
    if len(valid):
        conf = pd.crosstab(valid["llm_label"], valid["judge_label"]).reindex(
            index=CLASSES, columns=CLASSES, fill_value=0)
        print("\nconfusion (rows = labeler, cols = judge):")
        print(conf)
    if issues:
        print(f"\ntop issue tags: {issues}")

    gold = golden_check(res)
    if gold:
        print(f"\ngolden-set check (n={gold['n']}): labeler_acc={gold['labeler_acc']}  judge_acc={gold['judge_acc']}")
        print("  -> if judge_acc is low, fix the rubric/judge before trusting its verdicts")
    else:
        print("\nno golden_set.csv found — adjudicate the review queue and start one")

    append_run_log({
        "run_id": args.run_id,
        "ts": now_iso(),
        "labeler_version": args.labeler_version,
        "judge_model": JUDGE_MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "n": len(res),
        "n_parse_err": n_err,
        "agreement": agreement,
        "kappa": kappa,
        "agr_positive": pca.get("positive"),
        "agr_negative": pca.get("negative"),
        "agr_unsure": pca.get("unsure"),
        "dist_labeler": json.dumps(label_dist(res["llm_label"])),
        "dist_judge": json.dumps(label_dist(valid["judge_label"])) if len(valid) else "{}",
        "top_issues": json.dumps(issues),
    })

    results_path = OUT_DIR / f"results_{args.run_id}.csv"
    res.to_csv(results_path, index=False)
    queue_path = write_review_queue(res, args.run_id)
    print(f"\nwrote: {results_path}")
    print(f"wrote: {queue_path}  <- adjudicate this, append decisions to {GOLDEN_PATH}")
    print(f"log:   {RUNS_LOG}  (use --compare-last-two after your next run)")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-label QA harness (blind re-annotation)")
    ap.add_argument("--csv", help="input CSV with columns text,llm_label")
    ap.add_argument("--rubric", default="judge_rubric_v1.md")
    ap.add_argument("--n-fresh", type=int, default=20, help="fresh examples per class per run")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--labeler-version", default="unversioned",
                    help="version tag of the labeler prompt that produced llm_label")
    ap.add_argument("--skip-pass2", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--compare-last-two", action="store_true",
                    help="print drift report between the two most recent runs and exit")
    ap.add_argument("--ingest-queue", metavar="QUEUE_CSV",
                    help="append filled human_label rows from a review queue to golden_set.csv and the fixed core, then exit")
    args = ap.parse_args()

    if args.ingest_queue:
        ingest_queue(args.ingest_queue)
        return
    if args.compare_last_two:
        compare_last_two()
        return
    if not args.csv:
        ap.error("--csv is required (or use --compare-last-two)")
    run(args)


if __name__ == "__main__":
    main()
