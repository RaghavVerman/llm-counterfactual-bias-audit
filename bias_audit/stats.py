"""Statistics for counterfactual group comparisons.

Design: each skeleton (counterfactual prompt set) is filled for every group and
sampled ``n`` times per group. Samples from one skeleton are correlated, so

* the effect is the mean *within-skeleton* difference between groups,
* p-values come from a permutation test that shuffles group labels only
  *within* each skeleton (stratified), and
* confidence intervals come from a cluster bootstrap over skeletons.

Holm-Bonferroni correction is applied across every (metric, group pair) test,
because an audit checks many metrics at once and some would look
"significant" by chance.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def stratified_permutation_test(
    values: np.ndarray, is_b: np.ndarray, strata: np.ndarray,
    n_perm: int = 10_000, seed: int = 0, chunk: int = 2_000,
) -> float:
    """Two-sided p-value for mean(B) - mean(A), permuting labels within strata."""
    values = np.asarray(values, float)
    is_b = np.asarray(is_b, bool)
    codes = pd.factorize(np.asarray(strata))[0]
    order = np.argsort(codes, kind="stable")
    y, lab, codes = values[order], is_b[order], codes[order]

    def stat(labels):  # labels: (..., n) bool
        nb = labels.sum(-1)
        return (labels * y).sum(-1) / nb - ((~labels) * y).sum(-1) / (labels.shape[-1] - nb)

    observed = abs(stat(lab))
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        # Sorting by (stratum code + uniform noise) permutes positions within each stratum.
        perm = np.argsort(codes[None, :] + rng.random((k, len(y))), axis=1)
        hits += int((np.abs(stat(lab[perm])) >= observed - 1e-12).sum())
        done += k
    return (hits + 1) / (n_perm + 1)


def cluster_bootstrap_ci(
    per_stratum_diff: np.ndarray, n_boot: int = 5_000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float]:
    d = np.asarray(per_stratum_diff, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    sd = pooled_sd(a, b)
    return float((np.mean(b) - np.mean(a)) / sd) if sd > 0 else 0.0


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def compare_groups(
    df: pd.DataFrame, metrics: list[str], group_col: str = "group",
    stratum_col: str = "skeleton_id", alpha: float = 0.05,
    n_perm: int = 10_000, n_boot: int = 5_000, seed: int = 0,
) -> pd.DataFrame:
    """One row per (metric, group pair) with effect, CI, effect size and p-values.

    ``diff`` is always ``mean(group_b) - mean(group_a)``.
    """
    groups = list(dict.fromkeys(df[group_col]))
    rows = []
    for metric in metrics:
        for ga, gb in combinations(groups, 2):
            sub = df[df[group_col].isin([ga, gb])].dropna(subset=[metric])
            a = sub.loc[sub[group_col] == ga, metric].to_numpy()
            b = sub.loc[sub[group_col] == gb, metric].to_numpy()
            per = sub.groupby([stratum_col, group_col])[metric].mean().unstack(group_col).dropna()
            d_s = (per[gb] - per[ga]).to_numpy()
            if len(d_s) < 2:
                continue
            lo, hi = cluster_bootstrap_ci(d_s, n_boot=n_boot, alpha=alpha, seed=seed)
            p = stratified_permutation_test(
                sub[metric].to_numpy(), (sub[group_col] == gb).to_numpy(),
                sub[stratum_col].to_numpy(), n_perm=n_perm, seed=seed,
            )
            rows.append({
                "metric": metric, "group_a": ga, "group_b": gb,
                "mean_a": a.mean(), "mean_b": b.mean(), "diff": d_s.mean(),
                "ci_low": lo, "ci_high": hi, "cohens_d": cohens_d(a, b), "pooled_sd": pooled_sd(a, b),
                "frac_skeletons_b_higher": float((d_s > 0).mean()),
                "p_perm": p, "n_a": len(a), "n_b": len(b), "n_skeletons": len(d_s),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_perm"].to_numpy())
        out["significant"] = out["p_holm"] < alpha
    return out
