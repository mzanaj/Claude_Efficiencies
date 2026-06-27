#!/usr/bin/env python3
"""
text_eda.py — Exploratory data analysis + leakage detection for text classification data.

What it does, in order:
  1. Tabular sanity pass   — balance, length-by-class, empties, exact dups
  2. Label-conflict check  — same (normalized) text with >1 label
  3. Near-duplicate / leakage scan — semantic twins, and whether they straddle the split
  4. 2D projection         — UMAP of sentence embeddings, saved as interactive HTML
                             (View 1 colored by label = read OVERLAP;
                              View 2 colored by split = read LEAKAGE)
  5. Cluster x label cross-tab — finds inseparable regions worth manual inspection

It adapts to dataset size automatically (sampling for the plot, cluster count, etc.),
so you don't need to know your row count in advance.

Requirements:
    pip install pandas numpy scikit-learn sentence-transformers umap-learn plotly

Usage:
    Edit the CONFIG block below, then:  python text_eda.py
"""

import re
import sys
import numpy as np
import pandas as pd

# ----------------------------- CONFIG ------------------------------------- #
CSV_PATH   = "data.csv"     # path to your CSV
TEXT_COL   = "text"         # column holding the sentence
LABEL_COL  = "label"        # column holding the class label
SPLIT_COL  = "split"        # column with "train"/"test" (or None to skip leakage view)
MODEL_NAME = "all-MiniLM-L6-v2"  # any sentence-transformers model
DUP_SIM    = 0.95           # cosine similarity above which two rows are "near-duplicates"
OUT_DIR    = "."            # where the HTML plots get written
# -------------------------------------------------------------------------- #


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for exact/conflict matching."""
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    # ---- load -----------------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    for col in (TEXT_COL, LABEL_COL):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found. Available: {list(df.columns)}")
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df["_norm"] = df[TEXT_COL].map(normalize)
    n = len(df)
    print(f"Loaded {n:,} rows from {CSV_PATH}")

    has_split = SPLIT_COL is not None and SPLIT_COL in df.columns

    # ---- 1. tabular sanity pass ----------------------------------------
    section("1. Class balance")
    print(df[LABEL_COL].value_counts(normalize=True).round(4).to_string())

    section("2. Mean character length by class  (watch for length leakage)")
    print(df.groupby(LABEL_COL)[TEXT_COL].apply(lambda s: s.str.len().mean())
            .round(1).to_string())

    section("3. Empty / whitespace-only rows")
    empties = df[TEXT_COL].str.strip().eq("").sum()
    print(f"{empties} empty rows")

    section("4. Exact duplicates (after normalization)")
    dup_mask = df["_norm"].duplicated(keep=False)
    print(f"{dup_mask.sum()} rows participate in an exact-duplicate group "
          f"({df.loc[dup_mask, '_norm'].nunique()} distinct texts)")

    # ---- 2. label conflicts --------------------------------------------
    section("5. Label conflicts (same text, different labels — these HURT training)")
    conflicts = (df.groupby("_norm")[LABEL_COL].nunique()
                   .pipe(lambda s: s[s > 1]))
    if len(conflicts):
        print(f"{len(conflicts)} texts carry conflicting labels. Examples:")
        for norm_text in conflicts.index[:5]:
            labels = sorted(df.loc[df["_norm"] == norm_text, LABEL_COL].unique())
            sample = df.loc[df["_norm"] == norm_text, TEXT_COL].iloc[0]
            print(f"  labels={labels}  |  {sample[:80]}")
    else:
        print("None found.")

    # ---- embed (used by 3, 4, 5 below) ---------------------------------
    section("Embedding sentences  (this is the slow step)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        df[TEXT_COL].tolist(),
        normalize_embeddings=True,      # so dot product == cosine similarity
        batch_size=64,
        show_progress_bar=True,
    )

    # ---- 3. near-duplicate / leakage scan ------------------------------
    section("6. Near-duplicates and train/test leakage")
    from sklearn.neighbors import NearestNeighbors
    # brute cosine is fine to ~100k rows; swap in FAISS above that
    algo = "brute" if n <= 100_000 else "auto"
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm=algo).fit(emb)
    dist, idx = nn.kneighbors(emb)
    # column 0 is the point itself; column 1 is its nearest *other* neighbor
    sim = 1.0 - dist[:, 1]
    nbr = idx[:, 1]

    near_pairs = np.where(sim >= DUP_SIM)[0]
    seen, pairs = set(), []
    for i in near_pairs:
        j = nbr[i]
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    print(f"{len(pairs)} near-duplicate pairs at cosine >= {DUP_SIM}")

    if has_split and pairs:
        split = df[SPLIT_COL].values
        straddle = [(i, j) for i, j in pairs if split[i] != split[j]]
        print(f"  -> {len(straddle)} of them STRADDLE the split (train<->test leakage)")
        for i, j in straddle[:5]:
            print(f"     [{split[i]}] {df[TEXT_COL].iloc[i][:55]}")
            print(f"     [{split[j]}] {df[TEXT_COL].iloc[j][:55]}\n")
    elif not has_split:
        print("  (no split column set — skipping the leakage check)")

    # ---- 4. 2D projection ----------------------------------------------
    section("7. UMAP projection -> interactive HTML")
    import umap
    import plotly.express as px

    # plotly bogs down past ~30k points; sample for the *plot* only
    PLOT_CAP = 30_000
    if n > PLOT_CAP:
        plot_idx = np.random.RandomState(0).choice(n, PLOT_CAP, replace=False)
        print(f"  {n:,} rows > {PLOT_CAP:,}: sampling {PLOT_CAP:,} for the plot.")
    else:
        plot_idx = np.arange(n)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0)
    xy = reducer.fit_transform(emb[plot_idx])
    pdf = df.iloc[plot_idx].copy()
    pdf["x"], pdf["y"] = xy[:, 0], xy[:, 1]
    pdf["hover"] = pdf[TEXT_COL].str.slice(0, 120)

    render = "webgl" if len(pdf) > 5_000 else "auto"

    fig1 = px.scatter(pdf, x="x", y="y", color=pdf[LABEL_COL].astype(str),
                      hover_data={"hover": True, "x": False, "y": False},
                      opacity=0.6, render_mode=render,
                      title="Colored by LABEL — overlapping colors = ambiguous regions")
    fig1.write_html(f"{OUT_DIR}/projection_by_label.html")
    print(f"  wrote {OUT_DIR}/projection_by_label.html")

    if has_split:
        fig2 = px.scatter(pdf, x="x", y="y", color=pdf[SPLIT_COL].astype(str),
                          hover_data={"hover": True, "x": False, "y": False},
                          opacity=0.6, render_mode=render,
                          title="Colored by SPLIT — a tight cluster with both colors = leakage")
        fig2.write_html(f"{OUT_DIR}/projection_by_split.html")
        print(f"  wrote {OUT_DIR}/projection_by_split.html")

    # ---- 5. cluster x label cross-tab ----------------------------------
    section("8. Cluster x label cross-tab  (mixed rows = inseparable regions)")
    from sklearn.cluster import KMeans
    k = int(np.clip(round(np.sqrt(n / 2)), 5, 50))  # size-adaptive cluster count
    print(f"KMeans with k={k} (auto-chosen from dataset size)")
    df["_cluster"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
    ct = pd.crosstab(df["_cluster"], df[LABEL_COL])
    print(ct.to_string())
    # surface the most label-mixed clusters explicitly
    purity = ct.max(axis=1) / ct.sum(axis=1)
    worst = purity.sort_values().head(5)
    print("\nLeast 'pure' clusters (closest to a 50/50 mix — inspect these first):")
    print((worst.round(3)).to_string())

    section("Done")
    print("Open the two HTML files, then hover any point sitting deep in the "
          "wrong color to see whether it's a hard case or a mislabel.")


if __name__ == "__main__":
    main()
