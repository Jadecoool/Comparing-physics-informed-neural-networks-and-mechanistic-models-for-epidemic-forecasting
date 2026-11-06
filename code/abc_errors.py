"""
Distance functions for ABC-with-budget.

CONTRACT -- every error function has the signature

    f(obs, sim, **kwargs) -> (N,) float64

    obs : (m,) or (m, A)      observed weekly deaths over the scoring weeks
    sim : (N, m) or (N, m, A) the same weeks for each of N candidates

so it works unchanged for `target: total` and `target: by_age`. Register a new
one with the @register decorator and refer to it by name from the yaml.

    >>> from abc_errors import get_error
    >>> f = get_error("biased_wmape")
    >>> err = f(obs_total[3:9], pool_deaths[:, 3:9])

`wmape` and `biased_wmape` are the two the paper uses. They are NOT on the same
scale and their values must not be compared with each other -- see the note in
`biased_wmape`.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

ERROR_FUNCS: Dict[str, Callable[..., np.ndarray]] = {}


def register(name: str):
    def deco(fn):
        ERROR_FUNCS[name] = fn
        return fn
    return deco


def get_error(name: str) -> Callable[..., np.ndarray]:
    if name not in ERROR_FUNCS:
        raise KeyError(f"unknown error {name!r}; available: {sorted(ERROR_FUNCS)}")
    return ERROR_FUNCS[name]


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _align(obs: np.ndarray, sim: np.ndarray):
    """Broadcast obs to sim's shape and return (residual, obs_b) as float64."""
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if sim.ndim != obs.ndim + 1:
        raise ValueError(f"sim must have one more axis than obs; got {sim.shape} vs {obs.shape}")
    obs_b = obs[None, ...]
    return sim - obs_b, obs_b            # residual > 0  <=>  OVER-prediction


def _harmonic_weights(m: int, offset: float = 1.0) -> np.ndarray:
    """
    Recency weights over m scoring weeks, chronological (most recent LAST):

        w_j = 1 / (m + offset - j)      j = 0 (oldest) .. m-1 (newest)

    Equivalently w = 1/(age + offset + 1) counting back from the newest week.
    Harmonic decay has a much longer tail than a geometric one: over an 18-week
    window the oldest week still carries ~2% of the total weight.

    TWO CONVENTIONS ARE IN USE, and they differ by exactly one in the denominator:

      offset = 1  newest week 1/2, then 1/3, 1/4 ... oldest 1/(m+1).
                  The default, and what `biased_wmape` and `biased_mape` use.
                  Every run up to and including seir_1M_0216_wmape_from0308 was
                  scored with it, so it must not change or those distances stop
                  reproducing.
      offset = 0  newest week 1, then 1/2, 1/3 ... oldest 1/m. This is the
                  convention in the collaborator's wmape_error(weighted=True),
                  written there as w_i = 1/(1 + n - i) with i counting 1..n.
                  It leans harder on the most recent week: over 6 points the
                  newest carries 40.8% of the weight rather than 31.4%.

    Both are anchored on the NEWEST point, so dropping leading weeks and
    renormalising gives the same thing as recomputing on what is left. That is
    not true for weeks dropped from the end.
    """
    j = np.arange(m, dtype=np.float64)
    return 1.0 / (m + offset - j)


def _log_shift(a: np.ndarray, shift: float) -> np.ndarray:
    """log(a + shift), using the accurate log1p path for the default shift=1."""
    return np.log1p(a) if shift == 1.0 else np.log(a + shift)


def _collapse(num: np.ndarray, den: np.ndarray, floor: float) -> np.ndarray:
    """Sum over every axis except the candidate axis and divide."""
    axes = tuple(range(1, num.ndim))
    return num.sum(axis=axes) / np.maximum(den.sum(axis=axes), floor)


# -----------------------------------------------------------------------------
# distances
# -----------------------------------------------------------------------------
@register("wmape")
def wmape(obs: np.ndarray, sim: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """
    Weighted mean absolute percentage error (a.k.a. WAPE / MAD-Mean ratio):

        wMAPE = sum_t |y_t - yhat_t| / sum_t y_t

    `floor` keeps the denominator finite when every scoring week has zero deaths.
    """
    resid, obs_b = _align(obs, sim)
    return _collapse(np.abs(resid), np.broadcast_to(obs_b, resid.shape), floor)


@register("biased_wmape")
def biased_wmape(obs: np.ndarray, sim: np.ndarray, shift: float = 1.0,
                 offset: float = 1.0) -> np.ndarray:
    """
    Recency-weighted mean absolute LOG-RATIO error.

        biased-wMAPE = sum_j w_j * |log y_j - log yhat_j| / sum_j w_j

        w_j = 1 / (m + 1 - j)      j = 0 (oldest) .. m-1 (newest)

    Three things distinguish it from `wmape`:

    1. LOG SCALE. |log y - log yhat| = |log(y/yhat)| measures the RATIO, not the
       absolute deviation. Being out by a factor of 2 costs 0.693 whether the
       week has 10 deaths or 1000. Plain wMAPE divides by sum(y), which the peak
       weeks dominate, so under wMAPE the low-count head and tail of the wave
       barely register; here every week counts equally in relative terms.

    2. RECENCY WEIGHTS, harmonic rather than geometric (see _harmonic_weights).
       "biased" refers to this bias towards recent weeks -- the metric is
       symmetric between over- and under-prediction, both in log space.

    3. DENOMINATOR is sum(w), not sum(w * y), so the result is a weighted MEAN of
       |log ratio| and reads directly: 0.1 ~ typically out by 10%, 0.693 ~ out by
       a factor of 2.

    NOT COMPARABLE WITH `wmape`. Different units. Forecast skill (WIS, coverage)
    is computed from the forecasts themselves and stays comparable across the
    two, but acceptance thresholds and error values do not.

    `shift` handles zeros, which log cannot take. The default 1.0 gives
    |log((y+1)/(yhat+1))|: a week observed 0 and predicted 0 scores 0, and large
    counts are unaffected (log 1358 vs log 1357 differ by 7e-4). It does damp
    ratio errors among very small counts -- observed 2 vs predicted 4 scores
    0.511 rather than 0.693 -- which is deliberate, since counts that small are
    within Poisson noise of each other. Set shift=0.0 for the literal formula;
    that is only safe when every observed AND simulated value is strictly
    positive, which is not the case for this pool.

    `offset` picks the weight convention, see _harmonic_weights. The default 1.0
    is what every System B run on the London data used (newest week 1/2). Pass
    0.0 for the collaborator's convention (newest week 1), which is what the
    _nz metrics use; abc_smc_synth_retro.py asks for that one.
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if sim.ndim != obs.ndim + 1:
        raise ValueError(f"sim must have one more axis than obs; got {sim.shape} vs {obs.shape}")

    d = np.abs(_log_shift(sim, shift) - _log_shift(obs[None, ...], shift))

    m = d.shape[1]
    w = _harmonic_weights(m, offset=offset).reshape((1, m) + (1,) * (d.ndim - 2))

    axes = tuple(range(1, d.ndim))
    # denominator is sum(w) per candidate; for target="by_age" the sum also runs
    # over the age axis, so scale by the number of age groups to keep it a mean
    n_extra = int(np.prod(d.shape[2:])) if d.ndim > 2 else 1
    return (w * d).sum(axis=axes) / (float(w.sum()) * n_extra)


@register("biased_mape")
def biased_mape(obs: np.ndarray, sim: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """
    Recency-weighted mean absolute PERCENTAGE error -- `mape` with the harmonic
    weights of `biased_wmape`, and no log anywhere.

        biased-MAPE = sum_j w_j * |y_j - yhat_j| / y_j  /  sum_j w_j

        w_j = 1 / (m + 1 - j)      j = 0 (oldest) .. m-1 (newest)

    Same weights as `biased_wmape`, same denominator sum(w), so it reads the same
    way -- a weighted MEAN of relative error, 0.3 ~ typically out by 30%. What it
    drops is the log:

      - `biased_wmape` measures |log((y+1)/(yhat+1))|, which is SYMMETRIC in the
        ratio. Predicting 0 against an observed 1357 costs log(1358) = 7.2.
      - this one measures |y - yhat| / y, which is NOT. Under-prediction is capped:
        predicting 0 costs exactly 1.0 no matter what was observed, while
        over-predicting by 10x costs 9.0. An all-zero forecast therefore scores
        exactly 1.0, so keep the acceptance threshold well under 1.

    Unlike `biased_wmape` there is no shift, so the observed weeks must not be
    zero -- `floor` only stops a division by zero, it does not make a zero week
    meaningful. That is why the run that uses this scores from 2020-03-15 (week 4)
    rather than from the anchor: weeks 0..3 have no deaths at all.

    NOT COMPARABLE WITH `biased_wmape`, whose values are mean |log ratio|.
    """
    resid, obs_b = _align(obs, sim)
    rel = np.abs(resid) / np.maximum(obs_b, floor)

    m = rel.shape[1]
    # j counts positions INSIDE the scoring slice, not absolute week numbers, so
    # the newest scored week always gets 1/2 whichever week of the grid it is.
    w = _harmonic_weights(m).reshape((1, m) + (1,) * (rel.ndim - 2))

    axes = tuple(range(1, rel.ndim))
    n_extra = int(np.prod(rel.shape[2:])) if rel.ndim > 2 else 1
    return (w * rel).sum(axis=axes) / (float(w.sum()) * n_extra)


@register("biased_wmape_nz")
def biased_wmape_nz(obs: np.ndarray, sim: np.ndarray,
                    shift: float = 1.0) -> np.ndarray:
    """
    `biased_wmape` with the zero-observed weeks DROPPED, and the collaborator's
    weighting (offset=0, newest week 1).

        biased-wMAPE-nz = sum_k w_k * |log(y_k+s) - log(yhat_k+s)| / sum_k w_k

        k runs over the weeks with y > 0 only, oldest first, and
        w_k = 1 / (M - k) with M the number of SURVIVING weeks.

    The log is kept, so this stays SYMMETRIC between over- and under-prediction:
    an all-zero forecast costs the weighted mean of log(y+1), which for weeks
    4..22 of the London data is 3.4 to 6.5 -- far above what a real curve scores
    (0.30 to 0.46). That is the whole reason to keep the log. Every metric here
    without it charges at most 1.0 for predicting nothing, which turns "no
    epidemic" into a cheap answer and collapses the posterior.

    Dropping the zero weeks is a separate choice from the log. With shift=1 the
    log handles them perfectly well -- a week observed 0 and predicted 0 scores 0
    -- but it charges log(1+yhat) for any death at all in such a week, a one-sided
    constraint the null model can only satisfy by seeding late and then
    overshooting the peak. On the same pool, dropping weeks 0..3 moves the week-8
    median of the best 1000 from 131..10957 (wildly unstable) to a steady
    1500..1900, against an observed 1357.

    Every observed week being zero raises ValueError. Totals only.
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if obs.ndim != 1 or sim.ndim != 2:
        raise ValueError("biased_wmape_nz takes obs (m,) and sim (N, m); got "
                         f"{obs.shape} and {sim.shape}")

    keep = obs > 0
    if not keep.any():
        raise ValueError("every observed week is zero; nothing to score")

    y = obs[keep]
    yhat = sim[:, keep]
    d = np.abs(_log_shift(yhat, shift) - _log_shift(y[None, :], shift))

    w = _harmonic_weights(len(y), offset=0.0)
    return (w[None, :] * d).sum(axis=1) / w.sum()


@register("biased_mape_nz")
def biased_mape_nz(obs: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """
    `biased_mape` with the zero-observed weeks DROPPED rather than floored.

        biased-MAPE-nz = sum_k w_k * |y_k - yhat_k| / y_k  /  sum_k w_k

        k runs over the weeks with y > 0 only, oldest first, and
        w_k = 1 / (M - k) with M the number of SURVIVING weeks, i.e. the
        newest surviving week gets 1 and the oldest 1/M.

    So the weights are recomputed over what is left: with one surviving week that
    week is the whole distance, with two the newer gets 2/3 of the weight, and so
    on. This is the collaborator's weighting (offset=0), NOT the one the earlier
    runs used, so a biased_mape_nz value and a biased_wmape value are doubly
    incomparable. Dropping a week is NOT the same as flooring it -- flooring
    would charge
    |0 - yhat| / 1 = yhat, an unbounded penalty for any death at all in a week
    that recorded none.

    For London this leaves weeks 4..t of the 25-week grid, because weeks 0..3
    (2020-02-16 .. 2020-03-08) recorded no deaths and weeks 4..22 all did. The
    rule is written out in full anyway so it does not depend on that.

    Every observed week being zero raises ValueError: there is nothing to score.
    Totals only -- the by-age shape would need a per-age mask and is not used.
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if obs.ndim != 1 or sim.ndim != 2:
        raise ValueError("biased_mape_nz takes obs (m,) and sim (N, m); got "
                         f"{obs.shape} and {sim.shape}")

    keep = obs > 0
    if not keep.any():
        raise ValueError("every observed week is zero; nothing to score")

    y = obs[keep]
    yhat = sim[:, keep]
    rel = np.abs(yhat - y[None, :]) / y[None, :]

    # Weights over the SURVIVING weeks, newest last. offset=0 is the
    # collaborator's convention (newest gets 1, not 1/2) so that this metric and
    # her wmape_error(weighted=True) are the same number on the same input.
    w = _harmonic_weights(len(y), offset=0.0)
    return (w[None, :] * rel).sum(axis=1) / w.sum()


@register("mape")
def mape(obs: np.ndarray, sim: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Classic MAPE -- mean over weeks of |y - yhat| / y. Unstable near y = 0."""
    resid, obs_b = _align(obs, sim)
    axes = tuple(range(1, resid.ndim))
    return np.mean(np.abs(resid) / np.maximum(obs_b, floor), axis=axes)


@register("rmse")
def rmse(obs: np.ndarray, sim: np.ndarray) -> np.ndarray:
    resid, _ = _align(obs, sim)
    axes = tuple(range(1, resid.ndim))
    return np.sqrt(np.mean(resid ** 2, axis=axes))


@register("poisson_nll")
def poisson_nll(obs: np.ndarray, sim: np.ndarray, floor: float = 0.1) -> np.ndarray:
    """
    Negative Poisson log-likelihood (up to a constant). Not a wMAPE, but a useful
    sanity check that the ABC selection is not being driven by the metric alone.
    """
    resid, obs_b = _align(obs, sim)
    lam = np.maximum(resid + obs_b, floor)
    axes = tuple(range(1, resid.ndim))
    return np.sum(lam - obs_b * np.log(lam), axis=axes)


def _reference_wmape(real_data, simu_data):
    """Literal transcription of the paper's wmape(), for the self-test below."""
    return [np.sum(np.abs(real_data - simu)) / np.sum(np.abs(real_data))
            for simu in simu_data]


def _reference_biased_wmape(real_data, simu_data):
    """Literal transcription of the paper's biased_wmape(), for the self-test."""
    wmapes = []
    tn = len(simu_data[0])
    for i in range(len(simu_data)):
        numerator = 0.0
        denominator = 0.0
        for j in range(tn):
            wj = 1 / (tn + 1 - j)
            abs_log_diff = abs(np.log(real_data[j]) - np.log(simu_data[i][j]))
            numerator += wj * abs_log_diff
            denominator += wj
        wmapes.append(numerator / denominator if denominator != 0 else np.nan)
    return np.array(wmapes)


if __name__ == "__main__":
    obs = np.array([100.0, 200.0, 150.0, 50.0])
    sim = np.array([[100.0, 200.0, 150.0, 50.0],      # perfect
                    [110.0, 220.0, 165.0, 55.0],      # +10% everywhere
                    [90.0, 180.0, 135.0, 45.0]])      # -10% everywhere

    # --- the vectorised versions must reproduce the paper's loops exactly -----
    assert np.allclose(wmape(obs, sim), _reference_wmape(obs, sim)), "wmape drifted"
    assert np.allclose(biased_wmape(obs, sim, shift=0.0),
                       _reference_biased_wmape(obs, sim)), "biased_wmape drifted"
    print("OK: both match a literal transcription of the paper's code")

    print("\nwmape                 ", wmape(obs, sim))
    print("biased (shift=0, exact)", biased_wmape(obs, sim, shift=0.0))
    print("biased (shift=1, used) ", biased_wmape(obs, sim))

    # --- harmonic weights: newest 1/2, oldest 1/(m+1) -------------------------
    w = _harmonic_weights(4)
    assert np.allclose(w, [1/5, 1/4, 1/3, 1/2]), w
    print("\nweights m=4 (oldest->newest):", np.round(w / w.sum(), 3))
    w18 = _harmonic_weights(18)
    print("weights m=18 newest / oldest:",
          round(w18[-1] / w18.sum(), 3), "/", round(w18[0] / w18.sum(), 3))

    # --- zeros: the literal formula blows up, shift=1 does not ----------------
    obs0 = np.array([0.0, 37.0, 235.0])               # the real first weeks
    sim0 = np.array([[0.0, 37.0, 235.0],              # perfect, incl. the zero
                     [1.0, 37.0, 235.0],              # 1 death in the zero week
                     [0.0, 0.0, 0.0]])                # never got going
    with np.errstate(divide="ignore", invalid="ignore"):
        print("\nliteral  shift=0 :", biased_wmape(obs0, sim0, shift=0.0))
    print("used     shift=1 :", biased_wmape(obs0, sim0))
    assert np.isfinite(biased_wmape(obs0, sim0)).all(), "shift=1 must stay finite"
    assert biased_wmape(obs0, sim0)[0] == 0.0, "a perfect match must score 0"
    print("OK: shift=1 keeps zero weeks finite")
