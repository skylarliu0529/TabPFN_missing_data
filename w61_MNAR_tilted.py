#!/usr/bin/env python3
"""
Kim-Yu exponential-tilting MNAR benchmark for W61.

Default target: TRUMPISSUE1_W61, encoded as the ordinal scores 1..4:
  1 = All or nearly all issues
  2 = Many, but not all issues
  3 = A few issues
  4 = No or almost no issues

For each true gamma in --gammas, the script:
1. Fits a working linear model E(Y | X) on the full data using one-hot X.
2. Generates MNAR response indicators
       R_i ~ Bernoulli(expit(g_hat(x_i) - gamma * Y_i + delta)),
   where delta is calibrated to the target missingness rate.
3. Splits artificial nonrespondents into validation/follow-up and test sets.
4. Trains TabPFN on respondents only to estimate f_1(y | x).
5. Estimates gamma from validation nonrespondents using a TabPFN plug-in
   conditional mean and/or a mixed categorical Kim-Yu kernel conditional mean.
6. Compares four f_0(y | x) estimators: TabPFN-MNAR, TabPFN tilted with
   kernel gamma estimation, TabPFN tilted with TabPFN gamma estimation, and
   the original Kim-Yu kernel estimator.
"""

import argparse
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.optimize import brentq, minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.utils import check_random_state

from w61_bench import (
    BASE_SEED,
    DATA_PATH,
    DEFAULT_N_REPEATS,
    MICE_BOOT_DRAWS,
    NONRESPONSE_TOKENS,
    X_fixed,
    brier_multiclass,
    clean_categorical_series,
    encode_categories_with_missing_token,
    mean_wasserstein,
    normalize_token_set,
    safe_onehot_encoder,
    set_all_seeds,
)


DEFAULT_Y_COL = "TRUMPISSUE1_W61"
DEFAULT_GAMMAS = "1 0.5 0.25 0"
DEFAULT_MISSING_PROP = 0.10
DEFAULT_VALIDATION_PROP = 0.15
DEFAULT_H_QUANTILES = "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"
DEFAULT_OUT_DIR_PREFIX = "results_mnar_4methods_rep"
DEFAULT_GAMMA_ESTIMATOR = "all"

Y_LEVELS_TRUMPISSUE1 = [
    "All or nearly all issues",
    "Many, but not all issues",
    "A few issues",
    "No or almost no issues",
]

ORDINAL_X_ORDERS = {
    "F_AGECAT": ["18-29", "30-49", "50-64", "65+"],
    "F_EDUCCAT2": [
        "Less than high school",
        "High school graduate",
        "Some college, no degree",
        "Associate's degree",
        "College graduate/some post grad",
        "Postgraduate",
    ],
    "F_BORN": ["No", "Don't know/refused", "Yes"],
    "F_ATTEND": [
        "Never",
        "Seldom",
        "A few times a year",
        "Once or twice a month",
        "Once a week",
        "More than once a week",
    ],
    "F_INCOME_RECODE": ["<$30,000", "$30-$74,999", "$75,000+"],
    "F_REG": [
        "You are NOT registered to vote at your current address",
        "You are PROBABLY registered, but there is a chance your registration has lapsed",
        "You are ABSOLUTELY CERTAIN that you are registered to vote at your current address",
    ],
    "F_IDEO": [
        "Very liberal",
        "Liberal",
        "Moderate",
        "Conservative",
        "Very conservative",
    ],
    "F_VOLSUM": ["No", "Yes"],
    "F_EDUCCAT": ["H.S. graduate or less", "Some College", "College graduate+"],
    "F_INCOME": [
        "Less than $10,000",
        "$10,000 to less than $20,000",
        "$20,000 to less than $30,000",
        "$30,000 to less than $40,000",
        "$40,000 to less than $50,000",
        "$50,000 to less than $75,000",
        "$75,000 to less than $100,000",
        "$100,000 to less than $150,000",
        "$150,000 or more",
    ],
}


@dataclass
class PreparedData:
    y_col: str
    y_levels: list[str]
    y_int: np.ndarray
    y_score: np.ndarray
    x_filled: pd.DataFrame
    x_ohe: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-x))


def parse_float_list(text: str) -> list[float]:
    vals = []
    for raw in text.replace(",", " ").replace(":", " ").split():
        raw = raw.strip()
        if raw:
            vals.append(float(raw))
    if not vals:
        raise ValueError(f"No numeric values parsed from '{text}'.")
    return vals


def format_float_for_filename(x: float) -> str:
    return str(float(x)).replace("-", "neg")


def prepare_w61_data(y_col: str) -> PreparedData:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cannot find {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
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

    if y_col == "TRUMPISSUE1_W61":
        y_levels = [v for v in Y_LEVELS_TRUMPISSUE1 if v in set(df[y_col].dropna())]
        if len(y_levels) != 4:
            raise RuntimeError(
                f"Expected four TRUMPISSUE1 levels, found {len(y_levels)}: {y_levels}"
            )
    else:
        raise ValueError(
            "This tilted benchmark currently supports the ordinal target "
            "TRUMPISSUE1_W61. Add an explicit ordered Y_LEVELS list before using "
            f"{y_col}."
        )

    df = df[df[y_col].isin(y_levels)].copy().reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No rows left after filtering Y to substantive ordered levels.")

    y_cat = pd.Categorical(df[y_col], categories=y_levels, ordered=True)
    y_int = y_cat.codes.astype(int)
    y_score = y_int.astype(float) + 1.0

    x_raw = df[X_fixed].copy()
    x_filled = encode_categories_with_missing_token(x_raw, "__MISSING__")
    ohe = safe_onehot_encoder()
    x_ohe = ohe.fit_transform(x_filled)

    print(f"data path: {DATA_PATH}")
    print(f"Y_COL: {y_col}")
    print(f"n after substantive Y filtering: {len(df)}")
    print(f"Y_LEVELS: {y_levels}")
    print("\nY distribution:")
    vc = pd.Series(y_int).value_counts().sort_index()
    for k, cnt in vc.items():
        print(f"  score={k + 1:.0f} label={y_levels[k]} count={cnt} prop={cnt / len(y_int):.4f}")

    return PreparedData(
        y_col=y_col,
        y_levels=y_levels,
        y_int=y_int,
        y_score=y_score,
        x_filled=x_filled,
        x_ohe=x_ohe,
    )


def fit_working_linear_score(x_ohe: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    lr = LinearRegression()
    lr.fit(x_ohe, y_score)
    fitted = np.asarray(lr.predict(x_ohe), dtype=float)
    print("\nWorking linear model for simulation:")
    print(f"  fitted mean={np.mean(fitted):.4f} sd={np.std(fitted):.4f}")
    print(f"  fitted range=[{np.min(fitted):.4f}, {np.max(fitted):.4f}]")
    return fitted


def calibrate_delta_for_missingness(
    eta_without_delta: np.ndarray,
    missing_prop: float,
) -> float:
    target_response = 1.0 - missing_prop
    lo, hi = -50.0, 50.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        mean_response = float(np.mean(sigmoid(eta_without_delta + mid)))
        if mean_response < target_response:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def make_mnar_mask(
    linear_score: np.ndarray,
    y_score: np.ndarray,
    gamma_true: float,
    missing_prop: float,
    rng,
) -> tuple[np.ndarray, np.ndarray, float]:
    eta = linear_score - gamma_true * y_score
    delta = calibrate_delta_for_missingness(eta, missing_prop)
    response_prob = sigmoid(eta + delta)
    responded = rng.uniform(size=len(y_score)) < response_prob
    if np.all(responded) or not np.any(responded):
        raise RuntimeError("Degenerate response sample; adjust missing_prop or gamma grid.")
    return responded, response_prob, delta


def split_missing_validation_test(
    responded: np.ndarray,
    validation_prop: float,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    missing_idx = np.where(~responded)[0]
    if len(missing_idx) < 2:
        raise RuntimeError("Too few missing rows to split validation/test.")
    n_val = int(np.floor(validation_prop * len(missing_idx)))
    n_val = min(max(n_val, 1), len(missing_idx) - 1)
    val_idx = np.sort(rng.choice(missing_idx, size=n_val, replace=False))
    val_mask = np.zeros(len(responded), dtype=bool)
    val_mask[val_idx] = True
    test_mask = (~responded) & (~val_mask)
    return val_mask, test_mask


def build_feature_codes(x_filled: pd.DataFrame):
    codes = np.zeros((len(x_filled), x_filled.shape[1]), dtype=np.int32)
    ordinal_scores = {}
    missing_codes = {}

    for j, c in enumerate(x_filled.columns):
        values = x_filled[c].astype(str)
        categories = sorted(pd.unique(values))
        cat_to_code = {cat: idx for idx, cat in enumerate(categories)}
        codes[:, j] = values.map(cat_to_code).to_numpy(dtype=np.int32)
        missing_codes[j] = cat_to_code.get("__MISSING__")

        if c in ORDINAL_X_ORDERS:
            rank_map = {level: r for r, level in enumerate(ORDINAL_X_ORDERS[c])}
            ranks = np.full(len(categories), np.nan, dtype=np.float32)
            for cat, idx in cat_to_code.items():
                if cat in rank_map:
                    denom = max(len(ORDINAL_X_ORDERS[c]) - 1, 1)
                    ranks[idx] = rank_map[cat] / denom
            ordinal_scores[j] = ranks

    return codes, ordinal_scores, missing_codes


def mixed_distance_squared(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    ordinal_scores: dict[int, np.ndarray],
    missing_codes: dict[int, int | None],
    block_size: int = 512,
) -> np.ndarray:
    n_a, p = codes_a.shape
    n_b = codes_b.shape[0]
    out = np.zeros((n_a, n_b), dtype=np.float32)

    for start in range(0, n_a, block_size):
        end = min(start + block_size, n_a)
        block = np.zeros((end - start, n_b), dtype=np.float32)
        for j in range(p):
            a = codes_a[start:end, j]
            b = codes_b[:, j]
            if j in ordinal_scores:
                ranks = ordinal_scores[j]
                miss = missing_codes.get(j)
                ra = ranks[a]
                rb = ranks[b]
                dm = np.abs(ra[:, None] - rb[None, :])
                missing_a = np.isnan(ra)
                missing_b = np.isnan(rb)
                if miss is not None or np.any(missing_a) or np.any(missing_b):
                    both_missing = missing_a[:, None] & missing_b[None, :]
                    one_missing = missing_a[:, None] | missing_b[None, :]
                    dm[one_missing] = 1.0
                    dm[both_missing] = 0.0
            else:
                dm = (a[:, None] != b[None, :]).astype(np.float32)
            block += dm.astype(np.float32) ** 2
        out[start:end] = block / float(p)
    return out


def choose_bandwidth(
    d2_rr: np.ndarray,
    y_score_r: np.ndarray,
    quantiles: list[float],
) -> tuple[float, pd.DataFrame]:
    d_rr = np.sqrt(np.maximum(d2_rr, 0.0))
    nonzero = d_rr[d_rr > 0]
    if len(nonzero) == 0:
        raise RuntimeError("All respondent distances are zero; cannot choose bandwidth.")
    h_grid = np.unique(np.quantile(nonzero, quantiles))
    h_grid = h_grid[h_grid > 0]
    if len(h_grid) == 0:
        raise RuntimeError("Bandwidth grid is empty.")

    rows = []
    best_h = None
    best_sse = np.inf
    y = y_score_r.astype(float)
    mean_y = float(np.mean(y))
    for h in h_grid:
        k = np.exp(-d2_rr / (2.0 * h * h))
        np.fill_diagonal(k, 0.0)
        denom = k.sum(axis=1)
        pred = np.empty_like(y, dtype=float)
        good = denom > 1e-12
        pred[good] = (k[good] @ y) / denom[good]
        pred[~good] = mean_y
        sse = float(np.sum((y - pred) ** 2))
        rows.append({"h": float(h), "loo_sse": sse})
        if sse < best_sse:
            best_sse = sse
            best_h = float(h)
    return best_h, pd.DataFrame(rows)


def m0_kernel_mean(
    d2_vr: np.ndarray,
    y_score_r: np.ndarray,
    gamma: float,
    h: float,
) -> np.ndarray:
    k = np.exp(-d2_vr / (2.0 * h * h))
    tilt = np.exp(np.clip(gamma * y_score_r, -60.0, 60.0))
    weights = k * tilt[None, :]
    denom = weights.sum(axis=1)
    numer = weights @ y_score_r
    fallback = float(np.average(y_score_r, weights=tilt))
    out = np.empty(d2_vr.shape[0], dtype=float)
    good = denom > 1e-12
    out[good] = numer[good] / denom[good]
    out[~good] = fallback
    return out


def kernel_tilted_probabilities(
    d2_target_r: np.ndarray,
    y_int_r: np.ndarray,
    y_score_r: np.ndarray,
    gamma: float,
    h: float,
    k_classes: int,
) -> np.ndarray:
    k = np.exp(-d2_target_r / (2.0 * h * h))
    tilt = np.exp(np.clip(gamma * y_score_r, -60.0, 60.0))
    weights = k * tilt[None, :]
    denom = weights.sum(axis=1)

    prob = np.zeros((d2_target_r.shape[0], k_classes), dtype=float)
    for cls in range(k_classes):
        prob[:, cls] = weights[:, y_int_r == cls].sum(axis=1)

    fallback = np.bincount(y_int_r, weights=tilt, minlength=k_classes).astype(float)
    fallback_sum = fallback.sum()
    if fallback_sum <= 0:
        fallback[:] = 1.0 / k_classes
    else:
        fallback /= fallback_sum

    good = denom > 1e-12
    prob[good] /= denom[good, None]
    prob[~good] = fallback

    row_sum = prob.sum(axis=1)
    bad = row_sum <= 0
    if np.any(bad):
        prob[bad, :] = 1.0 / k_classes
        row_sum = prob.sum(axis=1)
    prob /= row_sum[:, None]
    return prob


def estimate_gamma_from_validation(
    d2_vr: np.ndarray,
    y_score_r: np.ndarray,
    y_score_v: np.ndarray,
    h: float,
    bracket: tuple[float, float],
) -> tuple[float, pd.DataFrame, str]:
    def estimating_sum(gamma: float) -> float:
        pred = m0_kernel_mean(d2_vr, y_score_r, gamma, h)
        return float(np.sum(y_score_v - pred))

    lo, hi = bracket
    rows = []
    grid = np.linspace(lo, hi, 61)
    for g in grid:
        val = estimating_sum(float(g))
        rows.append({"gamma": float(g), "G_gamma": val, "G2": val * val})
    trace = pd.DataFrame(rows)

    g_lo = estimating_sum(lo)
    g_hi = estimating_sum(hi)
    if np.sign(g_lo) == 0:
        return float(lo), trace, "root_boundary"
    if np.sign(g_hi) == 0:
        return float(hi), trace, "root_boundary"
    if np.sign(g_lo) != np.sign(g_hi):
        gamma_hat = brentq(estimating_sum, lo, hi, maxiter=200)
        return float(gamma_hat), trace, "brentq"

    res = minimize_scalar(
        lambda g: estimating_sum(float(g)) ** 2,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(res.x), trace, "minimize_G2"


def estimate_gamma_from_tabpfn_validation(
    prob_val: np.ndarray,
    y_score_v: np.ndarray,
    y_scores_by_class: np.ndarray,
    bracket: tuple[float, float],
) -> tuple[float, pd.DataFrame, str]:
    def estimating_sum(gamma: float) -> float:
        prob_tilt = tilt_probabilities(prob_val, y_scores_by_class, gamma)
        pred_mean = prob_tilt @ y_scores_by_class
        return float(np.sum(y_score_v - pred_mean))

    lo, hi = bracket
    rows = []
    grid = np.linspace(lo, hi, 61)
    for g in grid:
        val = estimating_sum(float(g))
        rows.append({"gamma": float(g), "G_gamma": val, "G2": val * val})
    trace = pd.DataFrame(rows)

    g_lo = estimating_sum(lo)
    g_hi = estimating_sum(hi)
    if np.sign(g_lo) == 0:
        return float(lo), trace, "tabpfn_root_boundary"
    if np.sign(g_hi) == 0:
        return float(hi), trace, "tabpfn_root_boundary"
    if np.sign(g_lo) != np.sign(g_hi):
        gamma_hat = brentq(estimating_sum, lo, hi, maxiter=200)
        return float(gamma_hat), trace, "tabpfn_brentq"

    res = minimize_scalar(
        lambda g: estimating_sum(float(g)) ** 2,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(res.x), trace, "tabpfn_minimize_G2"


def tilt_probabilities(prob: np.ndarray, y_scores_by_class: np.ndarray, gamma: float) -> np.ndarray:
    logits = np.log(np.clip(prob, 1e-12, 1.0)) + gamma * y_scores_by_class[None, :]
    logits = logits - logits.max(axis=1, keepdims=True)
    tilted = np.exp(logits)
    tilted /= tilted.sum(axis=1, keepdims=True)
    return tilted


def align_predict_proba(prob: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    out = np.zeros((prob.shape[0], k), dtype=float)
    for j, cls in enumerate(classes):
        cls_int = int(cls)
        if 0 <= cls_int < k:
            out[:, cls_int] = prob[:, j]
    row_sum = out.sum(axis=1)
    bad = row_sum <= 0
    if np.any(bad):
        out[bad, :] = 1.0 / k
        row_sum = out.sum(axis=1)
    out /= row_sum[:, None]
    return out


def fit_tabpfn_predict(
    x_filled: pd.DataFrame,
    y_int: np.ndarray,
    train_mask: np.ndarray,
    target_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_all_seeds(seed)
    import torch
    from tabpfn import TabPFNClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = x_filled.iloc[train_mask].copy()
    y_train = y_int[train_mask]
    x_test = x_filled.iloc[target_mask].copy()
    categorical_feature_indices = list(range(x_train.shape[1]))

    tabpfn = TabPFNClassifier(
        device=device,
        ignore_pretraining_limits=True,
        categorical_features_indices=categorical_feature_indices,
    )
    tabpfn.fit(x_train, y_train)
    prob_raw = tabpfn.predict_proba(x_test)
    classes = getattr(tabpfn, "classes_", np.arange(prob_raw.shape[1]))
    return align_predict_proba(prob_raw, classes, len(np.unique(y_int))), y_train


def summarize_prob_method(
    name: str,
    y_true_int: np.ndarray,
    prob: np.ndarray,
    labels: list[str],
    y_scores_by_class: np.ndarray,
    gamma_hat: float = np.nan,
    gamma_estimator: str = "",
    gamma_estimation_method: str = "",
) -> dict:
    k = len(labels)
    pred = prob.argmax(axis=1)
    acc = float(accuracy_score(y_true_int, pred))
    ll = float(log_loss(y_true_int, prob, labels=list(range(k))))
    br = float(brier_multiclass(y_true_int, prob, k))
    wd = float(mean_wasserstein(y_true_int, prob, k))
    cm = confusion_matrix(y_true_int, pred, labels=list(range(k)))
    avg_prob = prob.mean(axis=0)
    y_true_score = y_scores_by_class[y_true_int]
    m0_hat = prob @ y_scores_by_class
    observed_score_mean = float(np.mean(y_true_score))
    m0_hat_mean = float(np.mean(m0_hat))
    m0_error = m0_hat - y_true_score
    m0_bias = float(m0_hat_mean - observed_score_mean)
    m0_rmse = float(np.sqrt(np.mean(m0_error ** 2)))

    print(f"\n=== {name} ===")
    if np.isfinite(gamma_hat):
        print(f"Gamma_hat: {gamma_hat:.6f} ({gamma_estimator}/{gamma_estimation_method})")
    print(f"Accuracy: {acc:.4f}")
    print(f"LogLoss : {ll:.4f}")
    print(f"Brier   : {br:.4f}")
    print(f"WD      : {wd:.4f}")
    print(
        "m0 mean : "
        f"hat={m0_hat_mean:.4f} observed_score={observed_score_mean:.4f} "
        f"bias={m0_bias:.4f} rmse={m0_rmse:.4f}"
    )
    print("Average predicted prob over missing test:")
    for label, p in zip(labels, avg_prob):
        print(f"  {label}: {p:.4f}")
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels]))

    out = {
        "method": name,
        "gamma_hat": float(gamma_hat),
        "gamma_estimator": gamma_estimator,
        "gamma_estimation_method": gamma_estimation_method,
        "accuracy": acc,
        "logloss": ll,
        "brier": br,
        "wd": wd,
        "m0_hat": m0_hat_mean,
        "observed_score": observed_score_mean,
        "m0_bias": m0_bias,
        "m0_rmse": m0_rmse,
    }
    for label, p in zip(labels, avg_prob):
        out[f"avgprob_{label}"] = float(p)
    return out


def run_one_gamma_repeat(
    data: PreparedData,
    linear_score: np.ndarray,
    feature_codes: np.ndarray,
    ordinal_scores: dict[int, np.ndarray],
    missing_codes: dict[int, int | None],
    gamma_true: float,
    missing_prop: float,
    validation_prop: float,
    h_quantiles: list[float],
    gamma_bracket: tuple[float, float],
    gamma_estimator: str,
    seed: int,
    rep_index: int,
) -> tuple[list[dict], dict, pd.DataFrame, pd.DataFrame]:
    rng = check_random_state(seed)
    random.seed(seed)

    responded, response_prob, delta = make_mnar_mask(
        linear_score, data.y_score, gamma_true, missing_prop, rng
    )
    val_mask, test_mask = split_missing_validation_test(responded, validation_prop, rng)
    respondent_mask = responded

    print(
        f"respondents={respondent_mask.sum()} validation_missing={val_mask.sum()} "
        f"test_missing={test_mask.sum()} actual_missing={(~responded).mean():.4f}"
    )

    respondent_idx = np.where(respondent_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    target_mask = val_mask | test_mask
    val_pos = np.flatnonzero(val_mask[target_mask])
    test_pos = np.flatnonzero(test_mask[target_mask])
    y_scores_by_class = np.arange(1, len(data.y_levels) + 1, dtype=float)
    k_classes = len(data.y_levels)

    print("Training TabPFN on respondents and predicting validation/test missing cases...")
    prob_target, _ = fit_tabpfn_predict(
        data.x_filled,
        data.y_int,
        train_mask=respondent_mask,
        target_mask=target_mask,
        seed=seed,
    )
    prob_val = prob_target[val_pos]
    prob_mar = prob_target[test_pos]

    use_tabpfn_gamma = gamma_estimator in {"all", "tabpfn"}
    use_kernel_gamma = gamma_estimator in {"all", "kernel"}
    if not use_tabpfn_gamma and not use_kernel_gamma:
        raise ValueError(f"Unknown gamma_estimator: {gamma_estimator}")

    gamma_hat_tabpfn = np.nan
    gamma_method_tabpfn = ""
    gamma_trace_parts = []
    h_hat = np.nan
    h_trace = pd.DataFrame()
    gamma_hat_kernel = np.nan
    gamma_method_kernel = ""
    d2_tr = None

    if use_tabpfn_gamma:
        gamma_hat_tabpfn, gamma_trace_tabpfn, gamma_method_tabpfn = estimate_gamma_from_tabpfn_validation(
            prob_val,
            data.y_score[val_idx],
            y_scores_by_class,
            gamma_bracket,
        )
        gamma_trace_parts.append(
            gamma_trace_tabpfn.assign(
                gamma_true=gamma_true,
                rep=rep_index,
                gamma_estimator="tabpfn",
            )
        )
        print(f"Estimated gamma={gamma_hat_tabpfn:.6f} using {gamma_method_tabpfn}")

    if use_kernel_gamma:
        print("Computing respondent-respondent mixed distances for bandwidth CV...")
        d2_rr = mixed_distance_squared(
            feature_codes[respondent_idx],
            feature_codes[respondent_idx],
            ordinal_scores,
            missing_codes,
        )
        h_hat, h_trace = choose_bandwidth(d2_rr, data.y_score[respondent_idx], h_quantiles)
        print(f"Selected h={h_hat:.6f}")
        h_trace = h_trace.assign(
            gamma_true=gamma_true,
            rep=rep_index,
            gamma_estimator="kernel",
        )

        print("Computing validation-respondent mixed distances for gamma estimation...")
        d2_vr = mixed_distance_squared(
            feature_codes[val_idx],
            feature_codes[respondent_idx],
            ordinal_scores,
            missing_codes,
        )
        gamma_hat_kernel, gamma_trace_kernel, gamma_method_kernel = estimate_gamma_from_validation(
            d2_vr,
            data.y_score[respondent_idx],
            data.y_score[val_idx],
            h_hat,
            gamma_bracket,
        )
        gamma_trace_parts.append(
            gamma_trace_kernel.assign(
                gamma_true=gamma_true,
                rep=rep_index,
                gamma_estimator="kernel",
            )
        )
        print(f"Estimated gamma={gamma_hat_kernel:.6f} using {gamma_method_kernel}")

        print("Computing test-respondent mixed distances for Kim-Yu kernel prediction...")
        d2_tr = mixed_distance_squared(
            feature_codes[test_idx],
            feature_codes[respondent_idx],
            ordinal_scores,
            missing_codes,
        )

    y_test = data.y_int[test_mask]

    results = [
        summarize_prob_method(
            "TabPFN-MNAR",
            y_test,
            prob_mar,
            data.y_levels,
            y_scores_by_class,
            gamma_estimator="none",
            gamma_estimation_method="none",
        ),
    ]
    if use_kernel_gamma:
        prob_tilt_kernel = tilt_probabilities(prob_mar, y_scores_by_class, gamma_hat_kernel)
        results.append(
            summarize_prob_method(
                "TabPFN-tilted-MNAR-kernel-m0",
                y_test,
                prob_tilt_kernel,
                data.y_levels,
                y_scores_by_class,
                gamma_hat=gamma_hat_kernel,
                gamma_estimator="kernel",
                gamma_estimation_method=gamma_method_kernel,
            )
        )
    if use_tabpfn_gamma:
        prob_tilt_tabpfn = tilt_probabilities(prob_mar, y_scores_by_class, gamma_hat_tabpfn)
        results.append(
            summarize_prob_method(
                "TabPFN-tilted-MNAR-tabpfn-m0",
                y_test,
                prob_tilt_tabpfn,
                data.y_levels,
                y_scores_by_class,
                gamma_hat=gamma_hat_tabpfn,
                gamma_estimator="tabpfn",
                gamma_estimation_method=gamma_method_tabpfn,
            )
        )
    if use_kernel_gamma:
        prob_ky_kernel = kernel_tilted_probabilities(
            d2_tr,
            data.y_int[respondent_idx],
            data.y_score[respondent_idx],
            gamma_hat_kernel,
            h_hat,
            k_classes,
        )
        results.append(
            summarize_prob_method(
                "Kim-Yu-kernel-MNAR",
                y_test,
                prob_ky_kernel,
                data.y_levels,
                y_scores_by_class,
                gamma_hat=gamma_hat_kernel,
                gamma_estimator="kernel",
                gamma_estimation_method=gamma_method_kernel,
            )
        )

    metadata = {
        "gamma_true": gamma_true,
        "gamma_estimator_requested": gamma_estimator,
        "gamma_hat_kernel": gamma_hat_kernel,
        "gamma_estimation_method_kernel": gamma_method_kernel,
        "gamma_hat_tabpfn": gamma_hat_tabpfn,
        "gamma_estimation_method_tabpfn": gamma_method_tabpfn,
        "delta": delta,
        "n": len(data.y_int),
        "n_respondents": int(respondent_mask.sum()),
        "n_missing": int((~responded).sum()),
        "n_validation_missing": int(val_mask.sum()),
        "n_test_missing": int(test_mask.sum()),
        "actual_missing_prop": float((~responded).mean()),
        "mean_response_prob": float(response_prob.mean()),
        "respondent_y_mean": float(data.y_score[respondent_mask].mean()),
        "validation_y_mean": float(data.y_score[val_mask].mean()),
        "test_y_mean": float(data.y_score[test_mask].mean()),
    }

    gamma_trace = pd.concat(gamma_trace_parts, ignore_index=True) if gamma_trace_parts else pd.DataFrame()

    return results, metadata, h_trace, gamma_trace


def aggregate_results(
    y_col: str,
    gamma_true: float,
    repeat_results: list[list[dict]],
    metadata_rows: list[dict],
) -> pd.DataFrame:
    methods = []
    avgprob_keys = []
    for rep in repeat_results:
        for r in rep:
            if r["method"] not in methods:
                methods.append(r["method"])
            for key in r:
                if key.startswith("avgprob_") and key not in avgprob_keys:
                    avgprob_keys.append(key)

    rows = []
    for method in methods:
        runs = [r for rep in repeat_results for r in rep if r["method"] == method]
        n_runs = len(runs)
        gamma_vals = np.array([r.get("gamma_hat", np.nan) for r in runs], dtype=float)
        gamma_hat_mean = float(np.nanmean(gamma_vals)) if np.any(np.isfinite(gamma_vals)) else np.nan
        gamma_hat_se = (
            float(np.nanstd(gamma_vals, ddof=1) / np.sqrt(np.isfinite(gamma_vals).sum()))
            if np.isfinite(gamma_vals).sum() > 1
            else 0.0 if np.isfinite(gamma_vals).sum() == 1 else np.nan
        )
        gamma_estimators = sorted({r.get("gamma_estimator", "") for r in runs if r.get("gamma_estimator", "")})
        gamma_methods = sorted({r.get("gamma_estimation_method", "") for r in runs if r.get("gamma_estimation_method", "")})
        row = {
            "y_col": y_col,
            "gamma_true": gamma_true,
            "method": method,
            "n_repeats": n_runs,
            "gamma_estimator": ";".join(gamma_estimators),
            "gamma_estimation_method": ";".join(gamma_methods),
            "gamma_hat_mean": gamma_hat_mean,
            "gamma_hat_se": gamma_hat_se,
            "missing_prop_mean": float(np.mean([m["actual_missing_prop"] for m in metadata_rows])),
        }
        for metric in [
            "accuracy",
            "logloss",
            "brier",
            "wd",
            "m0_hat",
            "observed_score",
            "m0_bias",
            "m0_rmse",
        ]:
            vals = np.array([r[metric] for r in runs])
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_se"] = float(np.std(vals, ddof=1) / np.sqrt(n_runs)) if n_runs > 1 else 0.0
        for key in avgprob_keys:
            vals = np.array([r.get(key, np.nan) for r in runs])
            row[key] = float(np.nanmean(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kim-Yu tilted TabPFN MNAR benchmark for W61.")
    parser.add_argument("--y_col", default=DEFAULT_Y_COL, help="Ordinal target column. Default: TRUMPISSUE1_W61.")
    parser.add_argument("--gammas", default=DEFAULT_GAMMAS, help="Space- or comma-separated true gamma simulation values.")
    parser.add_argument("--n_repeats", type=int, default=int(os.environ.get("N_REPEATS", DEFAULT_N_REPEATS)))
    parser.add_argument("--missing_prop", type=float, default=float(os.environ.get("MISSING_PROP", DEFAULT_MISSING_PROP)))
    parser.add_argument("--validation_prop", type=float, default=float(os.environ.get("VALIDATION_PROP", DEFAULT_VALIDATION_PROP)))
    parser.add_argument(
        "--gamma_estimator",
        choices=["all", "tabpfn", "kernel"],
        default=os.environ.get("GAMMA_ESTIMATOR", DEFAULT_GAMMA_ESTIMATOR),
        help="Which gamma estimator/method subset to run. Default: all.",
    )
    parser.add_argument("--h_quantiles", default=DEFAULT_H_QUANTILES, help="Comma-separated distance quantiles for bandwidth grid.")
    parser.add_argument("--gamma_bracket", default="-3,3", help="Lower,upper bracket for gamma root search.")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    gammas = parse_float_list(args.gammas)
    h_quantiles = parse_float_list(args.h_quantiles)
    bracket_vals = parse_float_list(args.gamma_bracket)
    if len(bracket_vals) != 2:
        raise ValueError("--gamma_bracket must contain exactly two values, e.g. -3,3")
    gamma_bracket = (min(bracket_vals), max(bracket_vals))

    print(f"Y_COL           = {args.y_col}")
    print(f"GAMMAS          = {gammas}")
    print(f"N_REPEATS       = {args.n_repeats}")
    print(f"MISSING_PROP    = {args.missing_prop}")
    print(f"VALIDATION_PROP = {args.validation_prop}")
    print(f"GAMMA_ESTIMATOR = {args.gamma_estimator}")
    print(f"GAMMA_BRACKET   = {gamma_bracket}")

    data = prepare_w61_data(args.y_col)
    linear_score = fit_working_linear_score(data.x_ohe, data.y_score)
    feature_codes, ordinal_scores, missing_codes = build_feature_codes(data.x_filled)

    out_dir = args.out_dir or f"{DEFAULT_OUT_DIR_PREFIX}{args.n_repeats}"
    os.makedirs(out_dir, exist_ok=True)

    all_summaries = []
    all_metadata = []
    all_h_traces = []
    all_gamma_traces = []
    t0 = time.time()

    for gamma_true in gammas:
        repeat_results = []
        metadata_rows = []
        print("\n" + "#" * 70)
        print(f"# True gamma scenario: {gamma_true}")
        print("#" * 70)

        for rep in range(args.n_repeats):
            seed = BASE_SEED + rep
            print("\n" + "=" * 60)
            print(f"gamma={gamma_true} repeat={rep + 1}/{args.n_repeats} seed={seed}")
            print("=" * 60)
            rep_results, metadata, h_trace, gamma_trace = run_one_gamma_repeat(
                data=data,
                linear_score=linear_score,
                feature_codes=feature_codes,
                ordinal_scores=ordinal_scores,
                missing_codes=missing_codes,
                gamma_true=gamma_true,
                missing_prop=args.missing_prop,
                validation_prop=args.validation_prop,
                h_quantiles=h_quantiles,
                gamma_bracket=gamma_bracket,
                gamma_estimator=args.gamma_estimator,
                seed=seed,
                rep_index=rep,
            )
            repeat_results.append(rep_results)
            metadata_rows.append(metadata)
            all_metadata.append({"y_col": args.y_col, "rep": rep, **metadata})
            all_h_traces.append(h_trace)
            all_gamma_traces.append(gamma_trace)

        summary = aggregate_results(args.y_col, gamma_true, repeat_results, metadata_rows)
        all_summaries.append(summary)

    summary_all = pd.concat(all_summaries, ignore_index=True)
    metadata_all = pd.DataFrame(all_metadata)
    h_trace_all = pd.concat(all_h_traces, ignore_index=True) if all_h_traces else pd.DataFrame()
    gamma_trace_all = pd.concat(all_gamma_traces, ignore_index=True) if all_gamma_traces else pd.DataFrame()

    summary_path = os.path.join(out_dir, f"results_{args.y_col}_KY_Tilted.csv")
    metadata_path = os.path.join(out_dir, f"metadata_{args.y_col}_KY_Tilted.csv")
    h_trace_path = os.path.join(out_dir, f"bandwidth_trace_{args.y_col}_KY_Tilted.csv")
    gamma_trace_path = os.path.join(out_dir, f"gamma_trace_{args.y_col}_KY_Tilted.csv")

    summary_all.to_csv(summary_path, index=False)
    metadata_all.to_csv(metadata_path, index=False)
    h_trace_all.to_csv(h_trace_path, index=False)
    gamma_trace_all.to_csv(gamma_trace_path, index=False)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Saved {summary_path}")
    print(summary_all.to_string(index=False))
    print(f"Saved {metadata_path}")
    print(f"Saved {h_trace_path}")
    print(f"Saved {gamma_trace_path}")


if __name__ == "__main__":
    main()
