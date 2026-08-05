#!/usr/bin/env python3
"""
W61 MNAR benchmark.

This script keeps the data cleaning, feature encoding, model settings, and
metrics from w61_bench.py, but replaces the random stratified holdout with an
MNAR masking mechanism. The masked rows form the evaluation set and the
unmasked rows form the training set.

Available mechanisms:
- logistic_class: response probability with x_i^T beta + gamma Y_i + delta.
- logistic_xy: response probability from the screenshot,
      pi_i = exp(x_i^T beta + gamma X_i Y_i + delta)
             / (1 + exp(x_i^T beta + gamma X_i Y_i + delta)).
"""

import argparse
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

from w61_bench import (
    BASE_SEED,
    DATA_PATH,
    DEFAULT_N_REPEATS,
    DEFAULT_Y_COL,
    EXPLICIT_Y_LEVEL_ORDERS,
    MICE_BOOT_DRAWS,
    NONRESPONSE_TOKENS,
    RF_TREES,
    TABPFN_IGNORE_LIMITS,
    TABPFN_MAX_TRAIN,
    X_fixed,
    build_mixed_wd_distance_matrix,
    clean_categorical_series,
    encode_categories_with_missing_token,
    normalize_token_set,
    safe_onehot_encoder,
    set_all_seeds,
    stratified_bootstrap,
    summarize_method,
)


DEFAULT_MISSING_PROP = 0.10
DEFAULT_ALPHA = 0.50
DEFAULT_METHODS = "baseline,mice,rf,tabpfn"
DEFAULT_MECHANISM = "logistic_xy"
DEFAULT_Y_SCORE = "code"
RESULTS_PREFIX = "results_mnar_rep"

METHOD_ALIASES = {
    "baseline": "baseline",
    "majority": "baseline",
    "mice": "mice",
    "polyreg": "mice",
    "rf": "rf",
    "randomforest": "rf",
    "random_forest": "rf",
    "tabpfn": "tabpfn",
}

MECHANISM_ALIASES = {
    "logistic_class": "logistic_class",
    "class": "logistic_class",
    "logistic": "logistic_class",
    "logistic_xy": "logistic_xy",
    "xy": "logistic_xy",
    "screenshot": "logistic_xy",
}


@dataclass
class MaskingDesign:
    mechanism: str
    response_prob: np.ndarray
    metadata: pd.DataFrame
    delta: float | None = None
    gamma: float | None = None
    suffix: str = "MNAR"


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-x))


def parse_methods(methods_arg: str) -> list[str]:
    methods = []
    for raw in methods_arg.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in METHOD_ALIASES:
            raise ValueError(
                f"Unknown method '{raw}'. Use any of: {sorted(METHOD_ALIASES)}"
            )
        method = METHOD_ALIASES[key]
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("At least one method must be requested.")
    return methods


def parse_mechanism(mechanism_arg: str) -> str:
    key = mechanism_arg.strip().lower()
    if key not in MECHANISM_ALIASES:
        raise ValueError(
            f"Unknown mechanism '{mechanism_arg}'. "
            f"Use one of: {sorted(set(MECHANISM_ALIASES.values()))}"
        )
    return MECHANISM_ALIASES[key]


def format_float_for_filename(x: float) -> str:
    return str(float(x)).replace("-", "neg")


def standardize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = float(np.std(v))
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros_like(v, dtype=float)
    return (v - float(np.mean(v))) / sd


def align_predict_proba(prob: np.ndarray, classes: np.ndarray, K: int) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2:
        raise ValueError(f"predict_proba output must be 2-D, got shape {prob.shape}")
    if prob.shape[1] == K and np.array_equal(np.asarray(classes, dtype=int), np.arange(K)):
        return prob

    out = np.zeros((prob.shape[0], K), dtype=float)
    for j, cls in enumerate(classes):
        cls_int = int(cls)
        if 0 <= cls_int < K:
            out[:, cls_int] = prob[:, j]
    row_sum = out.sum(axis=1)
    bad = row_sum <= 0
    if np.any(bad):
        out[bad, :] = 1.0 / K
        row_sum = out.sum(axis=1)
    out /= row_sum[:, None]
    return out


def prepare_w61_data(y_col: str):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cannot find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"data path: {DATA_PATH}")
    print(f"Question: {y_col}")

    need = [y_col] + X_fixed
    missing_cols = [c for c in need if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    df = df[need].copy()

    normalized_nonresponse_tokens = normalize_token_set(NONRESPONSE_TOKENS)
    born_keep_tokens = normalize_token_set({"Don't know/refused"})
    born_nonresponse = normalized_nonresponse_tokens - born_keep_tokens

    df[y_col] = clean_categorical_series(df[y_col], normalized_nonresponse_tokens)
    for c in X_fixed:
        if c == "F_BORN":
            df[c] = clean_categorical_series(df[c], born_nonresponse)
        else:
            df[c] = clean_categorical_series(df[c], normalized_nonresponse_tokens)

    ordinal_order = [
        "All or nearly all issues",
        "Many, but not all issues",
        "A few issues",
        "No or almost no issues",
    ]
    all_vals = set(df[y_col].dropna().unique()) - normalized_nonresponse_tokens
    explicit_order = EXPLICIT_Y_LEVEL_ORDERS.get(y_col, [])
    y_levels = [v for v in explicit_order if v in all_vals]
    y_levels += [v for v in ordinal_order if v in all_vals and v not in y_levels]
    y_levels += sorted(all_vals - set(y_levels))

    print(f"Auto-detected Y_LEVELS ({len(y_levels)}): {y_levels}")
    wd_distance_matrix = build_mixed_wd_distance_matrix(y_col, y_levels)
    if wd_distance_matrix is not None:
        print("Using mixed-type WD distance matrix:")
        print(pd.DataFrame(wd_distance_matrix, index=y_levels, columns=y_levels).to_string())

    df = df[df[y_col].isin(y_levels)].copy().reset_index(drop=True)
    print("n after substantive Y filtering:", len(df))
    if len(df) == 0:
        raise RuntimeError("No rows left after filtering Y to Y_LEVELS.")

    y_cat = pd.Categorical(df[y_col], categories=y_levels, ordered=True)
    y_int = y_cat.codes.astype(int)
    if len(np.unique(y_int)) < 2:
        raise RuntimeError("Need at least two substantive target classes.")

    print("\nY distribution:")
    vc = pd.Series(y_int).value_counts().sort_index()
    for k, cnt in vc.items():
        print(f"  {k} ({y_levels[k]}): {cnt} ({cnt / len(y_int):.4f})")

    x_raw = df[X_fixed].copy()
    x_filled = encode_categories_with_missing_token(x_raw, "__MISSING__")
    ohe = safe_onehot_encoder()
    x_ohe = ohe.fit_transform(x_filled)

    return df, y_int, y_levels, x_ohe, x_filled, wd_distance_matrix


def fit_majority_linear_predictor(x_ohe: np.ndarray, y_int: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit the x_i^T beta score used by the older MNAR runs."""
    majority = pd.Series(y_int).value_counts().idxmax()
    y_majority = (y_int == majority).astype(int)
    lr = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        max_iter=4000,
    )
    lr.fit(x_ohe, y_majority)
    linear_pred = np.asarray(lr.decision_function(x_ohe), dtype=float)
    intercept = float(lr.intercept_[0])

    print("\nLinear predictor fitted from logistic regression:")
    print(f"  majority class = {majority}")
    print(f"  intercept = {intercept:.4f}")
    print(f"  std(linear_pred) = {np.std(linear_pred):.4f}")
    print(f"  range = [{np.min(linear_pred):.3f}, {np.max(linear_pred):.3f}]")
    return linear_pred, intercept


def calibrate_delta(eta_without_delta: np.ndarray, target_response_rate: float) -> float:
    if not (0.0 < target_response_rate < 1.0):
        raise ValueError("target_response_rate must be in (0, 1).")

    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        mean_pi = float(np.mean(sigmoid(eta_without_delta + mid)))
        if mean_pi < target_response_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def class_counts_and_prevalence(y_int: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y_int, minlength=K).astype(int)
    prevalence = counts.astype(float) / float(len(y_int))
    return counts, prevalence


def build_logistic_class_design(
    y_int: np.ndarray,
    y_levels: list[str],
    linear_pred: np.ndarray,
    missing_prop: float,
    alpha: float,
    gamma: float | None,
    y_score_mode: str,
) -> MaskingDesign:
    K = len(y_levels)
    _counts, prevalence = class_counts_and_prevalence(y_int, K)
    gamma_value = alpha if gamma is None else float(gamma)
    y_score_by_class, y_score_name = build_interaction_y_scores(
        y_int, prevalence, y_score_mode
    )

    eta = linear_pred + gamma_value * y_score_by_class[y_int]
    target_response_rate = 1.0 - missing_prop
    delta = calibrate_delta(eta, target_response_rate)
    response_prob = sigmoid(eta + delta)

    rows = []
    for k, label in enumerate(y_levels):
        idx = y_int == k
        rows.append(
            {
                "mechanism": "logistic_class",
                "alpha": alpha,
                "gamma": gamma_value,
                "delta": delta,
                "y_score": y_score_name,
                "label": label,
                "prevalence": prevalence[k],
                "y_score_k": y_score_by_class[k],
                "mean_pi_k": float(np.mean(response_prob[idx])),
                "mean_missing_rate_k": float(np.mean(1.0 - response_prob[idx])),
            }
        )

    suffix = (
        f"MNAR_logistic_class_alpha{format_float_for_filename(alpha)}"
        f"_gamma{format_float_for_filename(gamma_value)}"
    )
    return MaskingDesign(
        mechanism="logistic_class",
        response_prob=response_prob,
        metadata=pd.DataFrame(rows),
        delta=delta,
        gamma=gamma_value,
        suffix=suffix,
    )


def build_interaction_x(
    df: pd.DataFrame,
    linear_pred: np.ndarray,
    interaction_x_col: str | None,
) -> tuple[np.ndarray, str]:
    if interaction_x_col is None or interaction_x_col.strip() == "":
        return standardize(linear_pred), "standardized_linear_pred"

    if interaction_x_col not in df.columns:
        raise ValueError(
            f"--interaction_x_col '{interaction_x_col}' is not in the prepared data. "
            f"Available covariates include: {X_fixed}"
        )

    s = df[interaction_x_col].astype("object")
    s = s.where(s.notna(), "__MISSING__")
    categories = sorted(str(v) for v in pd.unique(s))
    cat = pd.Categorical(s.astype(str), categories=categories, ordered=True)
    return standardize(cat.codes.astype(float)), interaction_x_col


def build_interaction_y_scores(
    y_int: np.ndarray,
    prevalence: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, str]:
    mode = mode.strip().lower()
    if mode == "code":
        score_by_class = np.arange(len(prevalence), dtype=float)
    elif mode == "standardized_code":
        score_by_class = standardize(np.arange(len(prevalence), dtype=float))
    elif mode == "prevalence_log":
        ref = float(np.max(prevalence))
        score_by_class = np.log(np.clip(prevalence / ref, 1e-12, None))
    else:
        raise ValueError(
            "Y score mode must be 'code', 'standardized_code', or 'prevalence_log'."
        )

    if np.allclose(score_by_class, score_by_class[0]):
        score_by_class = np.zeros_like(score_by_class, dtype=float)
    return score_by_class.astype(float), mode


def build_logistic_xy_design(
    df: pd.DataFrame,
    y_int: np.ndarray,
    y_levels: list[str],
    linear_pred: np.ndarray,
    missing_prop: float,
    alpha: float,
    gamma: float | None,
    interaction_x_col: str | None,
    interaction_y_score: str,
) -> MaskingDesign:
    K = len(y_levels)
    counts, prevalence = class_counts_and_prevalence(y_int, K)
    gamma_value = alpha if gamma is None else float(gamma)

    x_interaction, x_source = build_interaction_x(df, linear_pred, interaction_x_col)
    y_score_by_class, y_score_mode = build_interaction_y_scores(
        y_int, prevalence, interaction_y_score
    )

    interaction = x_interaction * y_score_by_class[y_int]
    eta = linear_pred + gamma_value * interaction
    target_response_rate = 1.0 - missing_prop
    delta = calibrate_delta(eta, target_response_rate)
    response_prob = sigmoid(eta + delta)

    rows = []
    for k, label in enumerate(y_levels):
        idx = y_int == k
        rows.append(
            {
                "mechanism": "logistic_xy",
                "alpha": alpha,
                "gamma": gamma_value,
                "delta": delta,
                "interaction_x": x_source,
                "interaction_y_score": y_score_mode,
                "label": label,
                "prevalence": prevalence[k],
                "y_score_k": y_score_by_class[k],
                "mean_interaction_k": float(np.mean(interaction[idx])),
                "mean_pi_k": float(np.mean(response_prob[idx])),
                "mean_missing_rate_k": float(np.mean(1.0 - response_prob[idx])),
            }
        )

    suffix = (
        f"MNAR_logistic_xy_alpha{format_float_for_filename(alpha)}"
        f"_gamma{format_float_for_filename(gamma_value)}"
    )
    if x_source != "standardized_linear_pred":
        suffix += f"_x{x_source}"

    return MaskingDesign(
        mechanism="logistic_xy",
        response_prob=response_prob,
        metadata=pd.DataFrame(rows),
        delta=delta,
        gamma=gamma_value,
        suffix=suffix,
    )


def build_masking_design(
    mechanism: str,
    df: pd.DataFrame,
    y_int: np.ndarray,
    y_levels: list[str],
    linear_pred: np.ndarray,
    missing_prop: float,
    alpha: float,
    gamma: float | None,
    interaction_x_col: str | None,
    interaction_y_score: str,
    class_y_score: str,
) -> MaskingDesign:
    if mechanism == "logistic_class":
        return build_logistic_class_design(
            y_int,
            y_levels,
            linear_pred,
            missing_prop,
            alpha,
            gamma,
            class_y_score,
        )
    if mechanism == "logistic_xy":
        return build_logistic_xy_design(
            df,
            y_int,
            y_levels,
            linear_pred,
            missing_prop,
            alpha,
            gamma,
            interaction_x_col,
            interaction_y_score,
        )
    raise ValueError(f"Unhandled mechanism: {mechanism}")


def sample_mask_from_design(
    design: MaskingDesign,
    y_int: np.ndarray,
    rng,
) -> np.ndarray:
    n = len(y_int)
    mask = rng.uniform(size=n) > design.response_prob
    if not np.any(mask):
        mask[int(np.argmin(design.response_prob))] = True
    if np.all(mask):
        mask[int(np.argmax(design.response_prob))] = False
    return mask


def run_methods_for_mask(
    y_col: str,
    y_int: np.ndarray,
    y_levels: list[str],
    x_ohe: np.ndarray,
    x_filled: pd.DataFrame,
    mask: np.ndarray,
    seed: int,
    wd_distance_matrix: np.ndarray | None,
    methods: list[str],
    save_predictions: bool,
    out_dir: str,
    pred_tag: str,
    response_prob: np.ndarray,
    rep_index: int,
) -> list[dict]:
    rng = check_random_state(seed)
    set_all_seeds(seed)

    K = len(y_levels)
    train_mask = ~mask
    test_mask = mask

    x_train = x_ohe[train_mask]
    y_train = y_int[train_mask]
    x_test = x_ohe[test_mask]
    y_test = y_int[test_mask]

    if len(y_test) == 0:
        raise RuntimeError("MNAR mask produced an empty evaluation set.")
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("MNAR mask left fewer than two classes in training.")

    print(
        f"train={x_train.shape[0]} test(masked)={x_test.shape[0]} classes={K}",
        flush=True,
    )

    results = []
    predictions_store = {}

    if "baseline" in methods:
        majority = pd.Series(y_train).value_counts().idxmax()
        prob_base = np.zeros((len(y_test), K), dtype=float)
        prob_base[:, majority] = 1.0
        results.append(
            summarize_method(
                "Baseline (majority)",
                y_test,
                prob_base,
                y_levels,
                wd_distance_matrix=wd_distance_matrix,
            )
        )
        if save_predictions:
            predictions_store["Baseline (majority)"] = prob_base

    if "mice" in methods:
        prob_mice = np.zeros((len(y_test), K), dtype=float)
        for _ in range(MICE_BOOT_DRAWS):
            xb, yb = stratified_bootstrap(x_train, y_train, rng)
            lr = LogisticRegression(
                solver="lbfgs",
                multi_class="multinomial",
                penalty="l2",
                C=1.0,
                max_iter=4000,
            )
            lr.fit(xb, yb)
            prob_b = align_predict_proba(lr.predict_proba(x_test), lr.classes_, K)
            prob_mice += prob_b
        prob_mice /= float(MICE_BOOT_DRAWS)
        results.append(
            summarize_method(
                "MICE(polyreg)-bootstrap",
                y_test,
                prob_mice,
                y_levels,
                wd_distance_matrix=wd_distance_matrix,
            )
        )
        if save_predictions:
            predictions_store["MICE(polyreg)-bootstrap"] = prob_mice

    if "rf" in methods:
        rf = RandomForestClassifier(
            n_estimators=RF_TREES,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
        )
        rf.fit(x_train, y_train)
        prob_rf = align_predict_proba(rf.predict_proba(x_test), rf.classes_, K)
        results.append(
            summarize_method(
                "RandomForest",
                y_test,
                prob_rf,
                y_levels,
                wd_distance_matrix=wd_distance_matrix,
            )
        )
        if save_predictions:
            predictions_store["RandomForest"] = prob_rf

    if "tabpfn" in methods:
        try:
            import torch
            from tabpfn import TabPFNClassifier

            device = "cuda" if torch.cuda.is_available() else "cpu"
            x_tab_all = x_filled.copy()
            x_tab_train = x_tab_all.iloc[train_mask].copy()
            y_tab_train = y_int[train_mask]
            x_tab_test = x_tab_all.iloc[test_mask].copy()
            categorical_feature_indices = list(range(x_tab_train.shape[1]))

            if TABPFN_MAX_TRAIN is not None and x_tab_train.shape[0] > TABPFN_MAX_TRAIN:
                sub = []
                for k in range(K):
                    idx_k = np.where(y_tab_train == k)[0]
                    if len(idx_k) == 0:
                        continue
                    m = max(1, int(np.floor(TABPFN_MAX_TRAIN * (len(idx_k) / len(y_tab_train)))))
                    sub.append(rng.choice(idx_k, size=min(m, len(idx_k)), replace=False))
                sub = np.concatenate(sub)
                rng.shuffle(sub)
                x_tab_train = x_tab_train.iloc[sub].copy()
                y_tab_train = y_tab_train[sub]

            tabpfn = TabPFNClassifier(
                device=device,
                ignore_pretraining_limits=TABPFN_IGNORE_LIMITS,
                categorical_features_indices=categorical_feature_indices,
            )
            tabpfn.fit(x_tab_train, y_tab_train)
            prob_tabpfn_raw = tabpfn.predict_proba(x_tab_test)
            classes = getattr(tabpfn, "classes_", np.arange(prob_tabpfn_raw.shape[1]))
            prob_tabpfn = align_predict_proba(prob_tabpfn_raw, classes, K)
            results.append(
                summarize_method(
                    "TabPFN",
                    y_test,
                    prob_tabpfn,
                    y_levels,
                    wd_distance_matrix=wd_distance_matrix,
                )
            )
            if save_predictions:
                predictions_store["TabPFN"] = prob_tabpfn
        except Exception as e:
            print(f"\n[TabPFN skipped due to error] {repr(e)}")

    if save_predictions:
        os.makedirs(out_dir, exist_ok=True)
        flat_pred = {
            "y_test": y_test,
            "labels": np.array(y_levels),
            "mask_idx": np.where(test_mask)[0],
            "response_prob_test": response_prob[test_mask],
        }
        for method_name, prob_array in predictions_store.items():
            flat_pred[f"{method_name}__prob"] = prob_array
        pred_file = os.path.join(out_dir, f"predictions_rep{rep_index}_{y_col}_{pred_tag}.npz")
        np.savez(pred_file, **flat_pred)
        print(f"Saved predictions to {pred_file}")

    return results


def aggregate_results(
    y_col: str,
    all_repeat_results: list[list[dict]],
    missing_props: list[float],
    design: MaskingDesign,
    alpha: float,
) -> pd.DataFrame:
    method_names = []
    avgprob_keys = []
    for rep_results in all_repeat_results:
        for r in rep_results:
            if r["method"] not in method_names:
                method_names.append(r["method"])
            for k in r:
                if k.startswith("avgprob_") and k not in avgprob_keys:
                    avgprob_keys.append(k)

    metric_keys = ["accuracy", "logloss", "brier", "wd"]
    summary_rows = []
    for mname in method_names:
        method_runs = [
            r
            for rep_results in all_repeat_results
            for r in rep_results
            if r["method"] == mname
        ]
        n_runs = len(method_runs)
        if n_runs == 0:
            continue

        row = {
            "y_col": y_col,
            "mechanism": design.mechanism,
            "alpha": alpha,
            "gamma": design.gamma,
            "delta": design.delta,
            "method": mname,
            "n_repeats": n_runs,
            "missing_prop": float(np.mean(missing_props)),
        }
        for mk in metric_keys:
            vals = np.array([r[mk] for r in method_runs])
            row[f"{mk}_mean"] = float(np.mean(vals))
            row[f"{mk}_se"] = (
                float(np.std(vals, ddof=1) / np.sqrt(n_runs)) if n_runs > 1 else 0.0
            )
        for ak in avgprob_keys:
            vals = np.array([r.get(ak, np.nan) for r in method_runs])
            row[ak] = float(np.nanmean(vals))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_design_summary(design: MaskingDesign, missing_prop: float) -> None:
    print("\n" + "=" * 60)
    print(f"MNAR mechanism: {design.mechanism}")
    print(f"Target missing fraction = {missing_prop:.4f}")
    if design.gamma is not None:
        print(f"gamma = {design.gamma:.4f}")
    if design.delta is not None:
        print(f"delta = {design.delta:.4f}")
    print("=" * 60)
    print(design.metadata.to_string(index=False))
    print(
        f"Overall E[pi] = {np.mean(design.response_prob):.4f} "
        f"(target response rate = {1.0 - missing_prop:.4f})"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark W61 survey prediction under MNAR masking.")
    parser.add_argument(
        "--y_col",
        type=str,
        default=os.environ.get("Y_COL", DEFAULT_Y_COL),
        help="Target column name.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=int(os.environ.get("N_REPEATS", DEFAULT_N_REPEATS)),
        help="Number of repeated MNAR masks.",
    )
    parser.add_argument(
        "--missing_prop",
        type=float,
        default=float(os.environ.get("MISSING_PROP", DEFAULT_MISSING_PROP)),
        help="Target masked fraction.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=float(os.environ.get("ALPHA", DEFAULT_ALPHA)),
        help="MNAR strength. For logistic_class/logistic_xy, used as gamma unless --gamma is set.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Coefficient gamma for logistic_class/logistic_xy. Defaults to --alpha.",
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default=os.environ.get("MNAR_MECHANISM", DEFAULT_MECHANISM),
        help="MNAR mechanism: logistic_xy or logistic_class.",
    )
    parser.add_argument(
        "--interaction_x_col",
        type=str,
        default=os.environ.get("INTERACTION_X_COL", ""),
        help="Optional covariate column to use as scalar X_i in logistic_xy. "
        "Default uses standardized x_i^T beta.",
    )
    parser.add_argument(
        "--interaction_y_score",
        type=str,
        default=os.environ.get("INTERACTION_Y_SCORE", DEFAULT_Y_SCORE),
        help="Y_i score in logistic_xy: code, standardized_code, or prevalence_log.",
    )
    parser.add_argument(
        "--class_y_score",
        type=str,
        default=os.environ.get("CLASS_Y_SCORE", DEFAULT_Y_SCORE),
        help="Y_i score in logistic_class: code, standardized_code, or prevalence_log.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=os.environ.get("METHODS", DEFAULT_METHODS),
        help="Comma-separated methods: baseline,mice,rf,tabpfn.",
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Do not save per-repeat prediction npz files.",
    )
    args = parser.parse_args()

    y_col = args.y_col
    n_repeats = args.n_repeats
    missing_prop = args.missing_prop
    alpha = args.alpha
    gamma = args.gamma
    mechanism = parse_mechanism(args.mechanism)
    methods = parse_methods(args.methods)
    interaction_x_col = args.interaction_x_col.strip() or None

    print(f"Y_COL        = {y_col}")
    print(f"N_REPEATS    = {n_repeats}")
    print(f"MISSING_PROP = {missing_prop:.3f}")
    print(f"ALPHA        = {alpha:.3f}")
    print(f"GAMMA        = {gamma if gamma is not None else '<alpha>'}")
    print(f"MECHANISM    = {mechanism}")
    print(f"CLASS_Y      = {args.class_y_score}")
    print(f"INTERACT_Y   = {args.interaction_y_score}")
    print(f"METHODS      = {methods}")

    df, y_int, y_levels, x_ohe, x_filled, wd_distance_matrix = prepare_w61_data(y_col)
    linear_pred, _ = fit_majority_linear_predictor(x_ohe, y_int)

    design = build_masking_design(
        mechanism=mechanism,
        df=df,
        y_int=y_int,
        y_levels=y_levels,
        linear_pred=linear_pred,
        missing_prop=missing_prop,
        alpha=alpha,
        gamma=gamma,
        interaction_x_col=interaction_x_col,
        interaction_y_score=args.interaction_y_score,
        class_y_score=args.class_y_score,
    )
    print_design_summary(design, missing_prop)

    out_dir = f"{RESULTS_PREFIX}{n_repeats}"
    os.makedirs(out_dir, exist_ok=True)

    all_repeat_results = []
    missing_props = []
    t0 = time.time()

    for rep in range(n_repeats):
        seed = BASE_SEED + rep
        rng = check_random_state(seed)
        random.seed(seed)
        mask = sample_mask_from_design(design, y_int, rng)
        miss_prop_rep = float(np.mean(mask))
        missing_props.append(miss_prop_rep)

        print(f"\n{'=' * 50}")
        print(
            f"  Repeat {rep + 1}/{n_repeats}  "
            f"(seed={seed}, missing={miss_prop_rep:.4f})"
        )
        print(f"{'=' * 50}", flush=True)

        rep_results = run_methods_for_mask(
            y_col=y_col,
            y_int=y_int,
            y_levels=y_levels,
            x_ohe=x_ohe,
            x_filled=x_filled,
            mask=mask,
            seed=seed,
            wd_distance_matrix=wd_distance_matrix,
            methods=methods,
            save_predictions=not args.no_save_predictions,
            out_dir=out_dir,
            pred_tag=design.suffix,
            response_prob=design.response_prob,
            rep_index=rep,
        )
        all_repeat_results.append(rep_results)

    elapsed = time.time() - t0
    print(f"\nTotal runtime for {n_repeats} repeats: {elapsed:.1f}s")

    summary = aggregate_results(y_col, all_repeat_results, missing_props, design, alpha)
    results_path = os.path.join(out_dir, f"results_{y_col}_{design.suffix}.csv")
    masking_path = os.path.join(out_dir, f"masking_{y_col}_{design.suffix}.csv")

    summary.to_csv(results_path, index=False)
    design.metadata.to_csv(masking_path, index=False)

    print(f"\nSaved {results_path}")
    print(summary.to_string(index=False))
    print(f"\nSaved {masking_path}")


if __name__ == "__main__":
    main()
