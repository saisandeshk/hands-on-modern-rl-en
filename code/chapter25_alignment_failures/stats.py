"""Robust RL evaluation statistics from scratch (numpy only).

Implements the honest-reporting toolbox motivated by Section 25.5
(statistical rigor) and the rliable-style protocol: instead of reporting a
single seed or the best few runs, report the mean, median and interquartile
mean (IQM) together with stratified-bootstrap confidence intervals, and plot
a performance profile (fraction of runs above each score threshold) so that
cherry-picked conclusions become visible.

All randomness uses an explicit ``np.random.Generator`` so results are
reproducible from a single seed.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------
def mean_score(scores: np.ndarray) -> float:
    """Arithmetic mean of per-seed scores."""
    return float(np.mean(scores))


def median_score(scores: np.ndarray) -> float:
    """Median of per-seed scores (robust to a few lucky outliers)."""
    return float(np.median(scores))


def iqm_score(scores: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of sorted scores.

    The lowest 25% and highest 25% of runs are discarded before averaging,
    so one lucky or one catastrophic seed cannot dominate the headline
    number. On a fixed array this equals a direct "trim then average"
    computation, which the unit tests check exactly.
    """
    x = np.sort(np.asarray(scores, dtype=float).ravel())
    n = x.shape[0]
    if n == 0:
        raise ValueError("iqm_score requires at least one score")
    if n < 4:
        # Too few runs to trim quartiles; fall back to the plain mean.
        return float(np.mean(x))
    lo = n // 4
    hi = n - n // 4
    return float(np.mean(x[lo:hi]))


# ---------------------------------------------------------------------------
# Stratified bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(
    scores: np.ndarray,
    statistic=iqm_score,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for one group of seeds.

    Resamples ``scores`` with replacement ``n_resamples`` times, evaluates
    ``statistic`` on each resample, and returns ``(estimate, lower, upper)``
    where the interval covers the central ``ci`` mass of the bootstrap
    distribution (e.g. 2.5th/97.5th percentiles for 95%).
    """
    x = np.asarray(scores, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    estimate = float(statistic(x))
    boot = np.empty(n_resamples)
    for b in range(n_resamples):
        sample = rng.choice(x, size=x.shape[0], replace=True)
        boot[b] = statistic(sample)
    alpha = 1.0 - ci
    lower = float(np.percentile(boot, 100.0 * alpha / 2.0))
    upper = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return estimate, lower, upper


def stratified_bootstrap_ci(
    groups: dict[str, np.ndarray],
    statistic=iqm_score,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Stratified bootstrap: resample seeds *within* each group.

    "Stratified" means each group's resamples are drawn only from that
    group's own seeds, so groups with different seed counts stay comparable
    and no group's runs leak into another's interval. Returns a mapping
    ``group -> (estimate, lower, upper)``.
    """
    out: dict[str, tuple[float, float, float]] = {}
    for i, (name, scores) in enumerate(groups.items()):
        out[name] = bootstrap_ci(
            np.asarray(scores, dtype=float),
            statistic=statistic,
            n_resamples=n_resamples,
            ci=ci,
            seed=seed + i,
        )
    return out


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    statistic=iqm_score,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for ``statistic(a) - statistic(b)``.

    Each bootstrap replicate resamples both groups independently (paired by
    replicate index only for reproducibility), so the interval reflects seed
    noise on both sides. If the interval excludes zero, the ranking of the
    two configurations survives seed noise; otherwise the "winner" is not
    statistically resolved.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    estimate = float(statistic(a) - statistic(b))
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        boot[i] = statistic(rng.choice(a, size=a.shape[0], replace=True)) - statistic(
            rng.choice(b, size=b.shape[0], replace=True)
        )
    alpha = 1.0 - ci
    lower = float(np.percentile(boot, 100.0 * alpha / 2.0))
    upper = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return estimate, lower, upper


# ---------------------------------------------------------------------------
# Performance profiles
# ---------------------------------------------------------------------------
def performance_profile(
    groups: dict[str, np.ndarray], thresholds: np.ndarray
) -> dict[str, np.ndarray]:
    """Fraction of runs in each group scoring strictly above each threshold.

    For group ``g`` and threshold ``tau`` this is ``P(score > tau)`` over the
    group's seeds. A curve that stays above another across the whole
    threshold range is a robust win; curves that cross mean the ranking
    depends on the threshold -- exactly what a single headline number hides.
    """
    taus = np.asarray(thresholds, dtype=float)
    return {
        name: np.array([float(np.mean(np.asarray(s, dtype=float) > t)) for t in taus])
        for name, s in groups.items()
    }


def summarize(
    groups: dict[str, np.ndarray],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """One-row-per-group summary: mean, median, IQM and 95% IQM interval."""
    iqm_cis = stratified_bootstrap_ci(groups, statistic=iqm_score, n_resamples=n_resamples, seed=seed)
    rows: dict[str, dict[str, float]] = {}
    for name, scores in groups.items():
        x = np.asarray(scores, dtype=float)
        est, lo, hi = iqm_cis[name]
        rows[name] = {
            "mean": mean_score(x),
            "median": median_score(x),
            "iqm": est,
            "iqm_lo": lo,
            "iqm_hi": hi,
            "n": float(x.shape[0]),
        }
    return rows
