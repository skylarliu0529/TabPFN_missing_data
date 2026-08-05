#!/usr/bin/env python3
"""
W61 benchmark with supervised baselines and Wasserstein-based evaluation.

Method note (mixed-type WD):
- For selected targets (e.g., PPLFAV_c_W61), we compute Wasserstein-1 on a
    finite label space using a custom label-to-label ground distance matrix.
- Ordered labels use ordinal distances; nominal labels use fixed cross-type
    distances to ordered labels.
    
"""

import os
import sys
import time
import random
import re
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from sklearn.utils import check_random_state

# =========================
# Config
# =========================
DATA_PATH = "data/w61_party_filled.csv"
DEFAULT_Y_COL = "ELECT_TRUMP_W61"

X_fixed = [
    "F_METRO","F_CREGION","F_AGECAT","F_SEX","F_EDUCCAT2","F_RACETHN",
    "F_CITIZEN","F_NATIVITY2","F_MARITAL","F_RELIG",
    "F_BORN",
    "F_ATTEND",
    "F_RELTRAD","F_INCOME_RECODE","F_REG","F_IDEO",
    "F_ACSWEB","F_VOLSUM",
    "F_EDUCCAT", "F_HISP", "F_RACECMB", "F_INCOME", "F_PARTY_FINAL",
    "F_PARTYLN_FINAL"
    ## "F_PARTYSUM_FINAL"

]

# Y_LEVELS will be auto-generated from the data (see main())
Y_LEVELS = None

NONRESPONSE_TOKENS = {
    "Refused","Don't know/refused","DK/REF","Don't know","Dont know","DK",
    "No answer","Prefer not to answer","Skipped","Not asked","NA","N/A","",
}

# Optional explicit label order for specific targets.
# If provided, this order is used first, then any remaining labels are appended sorted.
EXPLICIT_Y_LEVEL_ORDERS = {
    "PPLFAV_c_W61": [
        "Very favorable",
        "Mostly favorable",
        "Mostly unfavorable",
        "Very unfavorable",
        "Never heard of this person",
    ],
}

# Optional mixed-type WD specifications.
# For PPLFAV: 4-level ordinal favorability + 1 nominal "Never heard" state.
MIXED_WD_SPECS = {
    "PPLFAV_c_W61": {
        "ordered_labels": [
            "Very favorable",
            "Mostly favorable",
            "Mostly unfavorable",
            "Very unfavorable",
        ],
        "nominal_labels": ["Never heard of this person"],
        # Distance between any nominal and any ordinal state.
        # Ordinal max distance above is 3, so 1.5 is middle penalty.
        "nominal_to_ordered_distance": 1.5,
    }
}

BASE_SEED = 2025
HOLDOUT_PROP = 0.10
DEFAULT_N_REPEATS = 10
MICE_BOOT_DRAWS = 30
RF_TREES = 800

TABPFN_IGNORE_LIMITS = True
TABPFN_MAX_TRAIN = None   

# =========================
# Utils
# =========================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def normalize_category_value(v):
    if pd.isna(v):
        return np.nan
    s = str(v).replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_token_set(tokens: set[str]) -> set[str]:
    return {normalize_category_value(t) for t in tokens}

def clean_categorical_series(s: pd.Series, nonresponse_tokens: set[str]) -> pd.Series:
    def _clean(v):
        nv = normalize_category_value(v)
        if pd.isna(nv) or nv in nonresponse_tokens:
            return np.nan
        return nv
    return s.apply(_clean)

def safe_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def brier_multiclass(y_true_int: np.ndarray, prob: np.ndarray, K: int) -> float:
    y_oh = np.eye(K)[y_true_int]
    return float(np.mean(np.sum((y_oh - prob) ** 2, axis=1)))

def build_mixed_wd_distance_matrix(y_col: str, labels: list[str]) -> np.ndarray | None:
    """
    Build a label-to-label ground distance matrix for mixed-type Wasserstein.
    Returns None when no mixed spec is defined for this y_col.
    """
    spec = MIXED_WD_SPECS.get(y_col)
    if spec is None:
        return None

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    K = len(labels)
    C = np.full((K, K), np.nan, dtype=float)
    np.fill_diagonal(C, 0.0)

    ordered = [lab for lab in spec["ordered_labels"] if lab in label_to_idx]
    nominal = [lab for lab in spec["nominal_labels"] if lab in label_to_idx]
    n2o = float(spec.get("nominal_to_ordered_distance", 1.0))

    # Ordinal block: absolute rank difference.
    for r1, l1 in enumerate(ordered):
        i = label_to_idx[l1]
        for r2, l2 in enumerate(ordered):
            j = label_to_idx[l2]
            C[i, j] = abs(r1 - r2)

    # Nominal <-> ordinal block: fixed cost.
    for ln in nominal:
        i = label_to_idx[ln]
        for lo in ordered:
            j = label_to_idx[lo]
            C[i, j] = n2o
            C[j, i] = n2o

    # Nominal block.
    for l1 in nominal:
        i = label_to_idx[l1]
        for l2 in nominal:
            j = label_to_idx[l2]
            C[i, j] = 0.0 if i == j else 1.0

    # Any leftover labels not covered by the spec: fallback to index distance.
    for i in range(K):
        for j in range(K):
            if np.isnan(C[i, j]):
                C[i, j] = abs(i - j)

    return C

def mean_wasserstein(
    y_true_int: np.ndarray,
    prob: np.ndarray,
    K: int,
    distance_matrix: np.ndarray | None = None,
) -> float:
    """
    Average 1-D Wasserstein distance (Earth Mover's Distance) over holdout rows.
    Uses scipy.stats.wasserstein_distance which computes:
        W_1(u, v) = integral |F_u(x) - F_v(x)| dx
    For each row: u = point mass at true class, v = predicted prob over {0..K-1}.
    """
    if distance_matrix is not None:
        # For a point mass at true class i and predicted distribution p,
        # Wasserstein-1 over a finite ground metric C reduces to E_j[C(i,j)].
        wd_sum = 0.0
        for r, i_true in enumerate(y_true_int):
            wd_sum += float(np.dot(distance_matrix[int(i_true)], prob[r]))
        return wd_sum / len(y_true_int)

    from scipy.stats import wasserstein_distance
    positions = np.arange(K)   # ordinal positions: 0, 1, ..., K-1
    wd_sum = 0.0
    for i in range(len(y_true_int)):
        # u_values = [true_class], u_weights = [1.0]  (point mass)
        # v_values = positions,     v_weights = prob[i] (predicted distribution)
        wd_sum += wasserstein_distance(
            [y_true_int[i]], positions,
            [1.0],           prob[i],
        )
    return wd_sum / len(y_true_int)

def stratified_holdout_indices(y_int: np.ndarray, holdout_prop: float, rng) -> np.ndarray:
    holdout = []
    for k in sorted(np.unique(y_int)):
        idx_k = np.where(y_int == k)[0]
        m = int(np.floor(len(idx_k) * holdout_prop))
        if m > 0:
            holdout.extend(rng.choice(idx_k, size=m, replace=False).tolist())
    return np.array(sorted(holdout), dtype=int)

def stratified_bootstrap(X: np.ndarray, y: np.ndarray, rng):
    rows = []
    for k in sorted(np.unique(y)):
        idx_k = np.where(y == k)[0]
        rows.append(rng.choice(idx_k, size=len(idx_k), replace=True))
    boot_idx = np.concatenate(rows)
    rng.shuffle(boot_idx)
    return X[boot_idx], y[boot_idx]

def summarize_method(
    name: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    labels: list[str],
    wd_distance_matrix: np.ndarray | None = None,
) -> dict:
    K = len(labels)
    y_pred = prob.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    ll = float(log_loss(y_true, prob, labels=list(range(K))))
    br = float(brier_multiclass(y_true, prob, K))
    wd = float(mean_wasserstein(y_true, prob, K, distance_matrix=wd_distance_matrix))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    avg_prob = prob.mean(axis=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"LogLoss : {ll:.4f}")
    print(f"Brier   : {br:.4f}")
    print(f"WD      : {wd:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])
    print(cm_df)
    print("\nAverage predicted prob over holdout:")
    for l, p in zip(labels, avg_prob):
        print(f"  {l}: {p:.4f}")

    out = {"method": name, "accuracy": acc, "logloss": ll, "brier": br, "wd": wd}
    for l, p in zip(labels, avg_prob):
        out[f"avgprob_{l}"] = float(p)
    return out

def encode_categories_with_missing_token(dfX: pd.DataFrame, missing_token="__MISSING__") -> pd.DataFrame:
    out = dfX.copy()
    for c in out.columns:
        out[c] = out[c].astype("object")
        out[c] = out[c].where(out[c].notna(), missing_token)
    return out

# =========================
# Single-repeat logic
# =========================
def run_one_repeat(Y_COL, df, X_fixed, y_int, Y_LEVELS, X_ohe, X_filled, seed, wd_distance_matrix, save_predictions=False, out_dir="results_rep10"):
    """Run one train/test split and return per-method metrics dict list, optionally save predictions."""
    rng = check_random_state(seed)
    set_all_seeds(seed)

    K = len(Y_LEVELS)

    holdout_idx = stratified_holdout_indices(y_int, HOLDOUT_PROP, rng)
    if len(holdout_idx) == 0:
        raise RuntimeError("Holdout set empty; increase HOLDOUT_PROP.")

    train_mask = np.ones(len(y_int), dtype=bool)
    train_mask[holdout_idx] = False

    X_train = X_ohe[train_mask]
    y_train = y_int[train_mask]
    X_test  = X_ohe[~train_mask]
    y_test  = y_int[~train_mask]

    print(f"\n--- seed={seed}  train={X_train.shape[0]}  test={X_test.shape[0]} ---", flush=True)

    results = []
    predictions_store = {}

    # Baseline (majority)
    majority = pd.Series(y_train).value_counts().idxmax()
    prob_base = np.zeros((len(y_test), K), dtype=float)
    prob_base[:, majority] = 1.0
    results.append(summarize_method("Baseline (majority)", y_test, prob_base, Y_LEVELS, wd_distance_matrix=wd_distance_matrix))
    if save_predictions:
        predictions_store["Baseline (majority)"] = prob_base

    # MICE-style polyreg baseline: bootstrap multinomial logistic regression
    prob_mice = np.zeros((len(y_test), K), dtype=float)
    for b in range(MICE_BOOT_DRAWS):
        Xb, yb = stratified_bootstrap(X_train, y_train, rng)
        lr = LogisticRegression(
            solver="lbfgs",
            multi_class="multinomial",
            penalty="l2",
            C=1.0,
            max_iter=4000,
        )
        lr.fit(Xb, yb)
        prob_mice += lr.predict_proba(X_test)
    prob_mice /= float(MICE_BOOT_DRAWS)
    results.append(summarize_method("MICE(polyreg)-bootstrap", y_test, prob_mice, Y_LEVELS, wd_distance_matrix=wd_distance_matrix))
    if save_predictions:
        predictions_store["MICE(polyreg)-bootstrap"] = prob_mice

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=RF_TREES,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
    )
    rf.fit(X_train, y_train)
    prob_rf = rf.predict_proba(X_test)
    results.append(summarize_method("RandomForest", y_test, prob_rf, Y_LEVELS, wd_distance_matrix=wd_distance_matrix))
    if save_predictions:
        predictions_store["RandomForest"] = prob_rf

    # TabPFN (supervised)
    try:
        import torch
        from tabpfn import TabPFNClassifier

        device = "cuda" if torch.cuda.is_available() else "cpu"

        X_tab_all = X_filled.copy()
        X_tab_train = X_tab_all.iloc[train_mask].copy()
        y_tab_train = y_int[train_mask]
        X_tab_test = X_tab_all.iloc[~train_mask].copy()
        categorical_feature_indices = list(range(X_tab_train.shape[1]))

        if TABPFN_MAX_TRAIN is not None and X_tab_train.shape[0] > TABPFN_MAX_TRAIN:
            sub = []
            for k in range(K):
                idx_k = np.where(y_tab_train == k)[0]
                m = max(1, int(np.floor(TABPFN_MAX_TRAIN * (len(idx_k) / len(y_tab_train)))))
                sub.append(rng.choice(idx_k, size=min(m, len(idx_k)), replace=False))
            sub = np.concatenate(sub)
            rng.shuffle(sub)
            X_tab_train = X_tab_train.iloc[sub].copy()
            y_tab_train = y_tab_train[sub]

        tabpfn = TabPFNClassifier(
            device=device,
            ignore_pretraining_limits=TABPFN_IGNORE_LIMITS,
            categorical_features_indices=categorical_feature_indices,
        )
        tabpfn.fit(X_tab_train, y_tab_train)
        prob_tabpfn = tabpfn.predict_proba(X_tab_test)
        results.append(summarize_method("TabPFN", y_test, prob_tabpfn, Y_LEVELS, wd_distance_matrix=wd_distance_matrix))
        if save_predictions:
            predictions_store["TabPFN"] = prob_tabpfn
    except Exception as e:
        print(f"\n[TabPFN skipped due to error] {repr(e)}")

    if save_predictions:
        os.makedirs(out_dir, exist_ok=True)
        # Flatten predictions: store y_test and labels once, methods as method__prob
        flat_pred = {"y_test": y_test, "labels": np.array(Y_LEVELS)}
        for method_name, prob_array in predictions_store.items():
            flat_pred[f"{method_name}__prob"] = prob_array
        pred_file = os.path.join(out_dir, f"predictions_{seed}_{Y_COL}_MAR.npz")
        np.savez(pred_file, **flat_pred)
        print(f"Saved predictions to {pred_file}")

    return results


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Benchmark classification methods on W61 survey data.")
    parser.add_argument("--y_col", type=str,
                        default=os.environ.get("Y_COL", DEFAULT_Y_COL),
                        help="Target column name (default: ELECT_TRUMP_W61)")
    parser.add_argument("--n_repeats", type=int,
                        default=int(os.environ.get("N_REPEATS", DEFAULT_N_REPEATS)),
                        help="Number of repeated holdout splits (default: 10)")
    args = parser.parse_args()

    Y_COL = args.y_col
    N_REPEATS = args.n_repeats

    print(f"Y_COL     = {Y_COL}")
    print(f"N_REPEATS = {N_REPEATS}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cannot find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"data path: {DATA_PATH}")
    print(f"Question: {Y_COL}")

    need = [Y_COL] + X_fixed
    missing_cols = [c for c in need if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    df = df[need].copy()

    # normalize text + token -> NaN
    normalized_nonresponse_tokens = normalize_token_set(NONRESPONSE_TOKENS)
    BORN_KEEP_TOKENS = normalize_token_set({"Don't know/refused"})
    BORN_NONRESPONSE = normalized_nonresponse_tokens - BORN_KEEP_TOKENS

    df[Y_COL] = clean_categorical_series(df[Y_COL], normalized_nonresponse_tokens)
    for c in X_fixed:
        if c == "F_BORN":
            df[c] = clean_categorical_series(df[c], BORN_NONRESPONSE)
        else:
            df[c] = clean_categorical_series(df[c], normalized_nonresponse_tokens)

    # Auto-generate Y_LEVELS from data
    ORDINAL_ORDER = [
        "All or nearly all issues",
        "Many, but not all issues",
        "A few issues",
        "No or almost no issues",
    ]
    global Y_LEVELS
    all_vals = set(df[Y_COL].dropna().unique()) - normalized_nonresponse_tokens
    explicit_order = EXPLICIT_Y_LEVEL_ORDERS.get(Y_COL, [])
    Y_LEVELS = [v for v in explicit_order if v in all_vals]
    Y_LEVELS += [v for v in ORDINAL_ORDER if v in all_vals and v not in Y_LEVELS]
    Y_LEVELS += sorted(all_vals - set(Y_LEVELS))
    print(f"Auto-detected Y_LEVELS ({len(Y_LEVELS)}): {Y_LEVELS}")

    wd_distance_matrix = build_mixed_wd_distance_matrix(Y_COL, Y_LEVELS)
    if wd_distance_matrix is not None:
        print("Using mixed-type WD distance matrix:")
        print(pd.DataFrame(wd_distance_matrix, index=Y_LEVELS, columns=Y_LEVELS).to_string())

    df = df[df[Y_COL].isin(Y_LEVELS)].copy().reset_index(drop=True)
    print("n after keeping substantive Y only:", len(df))
    if len(df) == 0:
        raise RuntimeError("No rows left after filtering Y to Y_LEVELS (check labels).")

    K = len(Y_LEVELS)
    y_cat = pd.Categorical(df[Y_COL], categories=Y_LEVELS, ordered=True)
    y_int = y_cat.codes.astype(int)

    print("\nY distribution:")
    vc = pd.Series(y_int).value_counts().sort_index()
    for k, cnt in vc.items():
        print(f"  {k} ({Y_LEVELS[k]}): {cnt}")

    X_raw = df[X_fixed].copy()
    X_filled = encode_categories_with_missing_token(X_raw, "__MISSING__")
    ohe = safe_onehot_encoder()
    X_ohe = ohe.fit_transform(X_filled)

    # Create output directory for results and predictions
    out_dir = f"results_rep{N_REPEATS}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Run multiple repeats ----
    all_repeat_results = []
    t0 = time.time()
    for rep in range(N_REPEATS):
        seed = BASE_SEED + rep
        print(f"\n{'='*50}")
        print(f"  Repeat {rep+1}/{N_REPEATS}  (seed={seed})")
        print(f"{'='*50}", flush=True)
        rep_results = run_one_repeat(Y_COL, df, X_fixed, y_int, Y_LEVELS,
                         X_ohe, X_filled, seed, wd_distance_matrix,
                         save_predictions=True, out_dir=out_dir)
        all_repeat_results.append(rep_results)

    elapsed = time.time() - t0
    print(f"\nTotal runtime for {N_REPEATS} repeats: {elapsed:.1f}s")

    # ---- Aggregate across repeats ----
    # Collect all method names that appeared in any repeat
    method_names = []
    for rep_results in all_repeat_results:
        for r in rep_results:
            if r["method"] not in method_names:
                method_names.append(r["method"])

    metric_keys = ["accuracy", "logloss", "brier", "wd"]
    avgprob_keys = [k for k in all_repeat_results[0][0] if k.startswith("avgprob_")]

    summary_rows = []
    for mname in method_names:
        # Gather this method's results across repeats
        method_runs = []
        for rep_results in all_repeat_results:
            for r in rep_results:
                if r["method"] == mname:
                    method_runs.append(r)
        n_runs = len(method_runs)
        if n_runs == 0:
            continue

        row = {"y_col": Y_COL, "method": mname, "n_repeats": n_runs}
        for mk in metric_keys:
            vals = np.array([r[mk] for r in method_runs])
            row[f"{mk}_mean"] = float(np.mean(vals))
            row[f"{mk}_se"] = float(np.std(vals, ddof=1) / np.sqrt(n_runs)) if n_runs > 1 else 0.0
        for ak in avgprob_keys:
            vals = np.array([r.get(ak, np.nan) for r in method_runs])
            row[ak] = float(np.nanmean(vals))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    out_path = os.path.join(out_dir, f"results_{Y_COL}.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
