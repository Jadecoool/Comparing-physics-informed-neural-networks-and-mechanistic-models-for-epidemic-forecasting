"""
Batch-vectorised age-structured stochastic SEIR model.

Same model as SEIR_model_general.SEIR(), but N candidate parameter sets are
advanced *simultaneously*: every compartment is an (N, A) integer array and each
binomial draw is a single numpy call on N*A elements instead of A scalar calls.

Measured on this machine (16 cores, 170-day simulation, 16 age groups):

    original SEIR()                 0.71   s/sample   ->  197 h for 1e6
    numba-jitted scalar loop        0.088  s/sample   ->   24 h for 1e6
    this batch implementation       0.016  s/sample   ->  4.6 h on 1 core
                                                          ~18 min on 16 cores

Model definition (unchanged from the original):
  * compartments S -> E -> I -> R, `daily_steps` tau-leap sub-steps per day
  * force of infection  lambda_k(t) = beta * sum_j C_kj(t) * I_j / N_j
  * beta = R0 * mu / rho(C_hat(day 0)),  C_hat_ij = (N_i/N_j) * C_ij
  * deaths on day (t-1)+Delta ~ Binomial(new recoveries on day t, ifr_k)
  * seeding: inf_t0 individuals split across age groups proportionally to N_k and
    across E/I proportionally to the latent/infectious period lengths

Two deliberate fixes relative to the original are flagged with "FIX:" below.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from abc_config import Setup


def simulate_batch(R0: np.ndarray,
                   inf_t0: np.ndarray,
                   Delta: np.ndarray,
                   setup: Setup,
                   rng: np.random.Generator,
                   want_deaths_by_age: bool = False,
                   want_infections: bool = False,
                   want_infections_by_age: bool = False,
                   want_S: bool = False,
                   seed_day: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Simulate `N = len(R0)` candidates at once.

    Parameters
    ----------
    R0, inf_t0 : (N,) float arrays. `inf_t0` is the number of individuals seeded
        at start_org_date; it was called `i0` until the parameter names were
        aligned with system B, which has always called it inf_t0.
    Delta  : (N,) int array, death-reporting delay in days
    setup  : resolved Setup (contact stack, IFR, population, weekly bins)
    rng    : numpy Generator; the whole batch shares one stream
    seed_day : (N,) int array or None. Day index (relative to start_org_date) on
        which each candidate's inf_t0 seeds are injected; everything stays fully
        susceptible before then. None (the default) means day 0 for everyone,
        which reproduces the original model exactly. Holding the seeds back by
        `d` days is exactly equivalent to starting that candidate's epidemic on
        `start_org_date + d`, so this is how a calibrated start date is
        implemented without having to vary the array length per candidate.

    Returns
    -------
    dict with, at minimum,
        "deaths_total" : (N, W) float32, weekly deaths from start_simu_date on
    and optionally
        "deaths_by_age"     : (N, W, A) float32
        "infections_total"  : (N, W)    float32
        "infections_by_age" : (N, W, A) float32
        "S_total"           : (N, W)    float32

    S_total is a STOCK, not a flow: it is the number of people still susceptible
    ON the day each weekly bin closes, not a sum over the week. It cannot be
    reconstructed afterwards from infections_total -- that array counts E->I
    transitions, while S falls on S->E, so cumsum(infections) lags the true
    depletion by whatever is sitting in E. And it cannot be recovered by
    re-simulating one candidate either: candidates share one RNG stream per chunk
    (see `replicate` below). Storing it while the pool is built is the only way.
    """
    R0 = np.asarray(R0, dtype=np.float64)
    inf_t0 = np.asarray(inf_t0, dtype=np.float64)
    Delta = np.asarray(Delta, dtype=np.int64)
    N = R0.shape[0]

    Nk = setup.Nk
    Cs = setup.Cs
    ifr = setup.ifr
    A, T = setup.A, setup.T
    eps, mu = setup.eps, setup.mu
    steps = setup.daily_steps
    dt = 1.0 / steps

    # ---- transmission rate ---------------------------------------------------
    beta = (R0 * mu / setup.eig0)[:, None]                       # (N, 1)

    # ---- seeding -------------------------------------------------------------
    frac = Nk / Nk.sum()
    denom = (1.0 / mu) + (1.0 / eps)
    # np.floor matches the original's .astype(int) truncation (all values >= 0)
    I0 = np.floor(inf_t0[:, None] * frac[None, :] * (1.0 / mu) / denom).astype(np.int64)
    E0 = np.floor(inf_t0[:, None] * frac[None, :] * (1.0 / eps) / denom).astype(np.int64)

    if seed_day is None:
        sday = np.zeros(N, dtype=np.int64)
    else:
        sday = np.asarray(seed_day, dtype=np.int64)
        if sday.shape != (N,):
            raise ValueError(f"seed_day must have shape ({N},), got {sday.shape}")
        if sday.min() < 0 or sday.max() > T - 2:
            raise ValueError(f"seed_day must lie in [0, {T-2}]; got "
                             f"[{sday.min()}, {sday.max()}]. Move start_org_date "
                             f"earlier so every candidate has room to be seeded.")

    # everyone starts fully susceptible; the seeds go in on each candidate's own
    # start day, inside the loop below (day 0 is injected on the first iteration,
    # which is exactly what the original model does)
    I = np.zeros((N, A), dtype=np.int64)
    E = np.zeros((N, A), dtype=np.int64)
    R = np.zeros((N, A), dtype=np.int64)
    S = np.broadcast_to(Nk.astype(np.int64), (N, A)).copy()

    # ---- output buffers (daily, aggregated to weeks at the end) --------------
    deaths_daily = np.zeros((N, T), dtype=np.float64)
    deaths_daily_age = np.zeros((N, T, A), dtype=np.float32) if want_deaths_by_age else None
    infec_daily = np.zeros((N, T), dtype=np.float64) if want_infections else None
    infec_daily_age = np.zeros((N, T, A), dtype=np.float32) if want_infections_by_age else None
    S_daily = np.zeros((N, T), dtype=np.float64) if want_S else None

    # transition probabilities that do not depend on the state
    pI = 1.0 - np.exp(-eps * dt)
    pR = 1.0 - np.exp(-mu * dt)

    row = np.arange(N)
    ifr_row = ifr[None, :]
    invN = 1.0 / Nk

    for t in range(1, T):
        # inject the candidates whose start day is t-1, so that the state going
        # into this day's dynamics is the seeded one
        inj = sday == (t - 1)
        if inj.any():
            E[inj] = E0[inj]
            I[inj] = I0[inj]
            S[inj] = Nk.astype(np.int64)[None, :] - E0[inj] - I0[inj]

        Ct = Cs[t].T.copy()               # (A, A), contiguous for the matmul
        new_R_day = np.zeros((N, A), dtype=np.int64)
        new_I_day = np.zeros((N, A), dtype=np.int64)

        for _ in range(steps):
            # lambda_k = beta * sum_j C_kj * I_j / N_j   ->  (I * invN) @ Ct
            foi = beta * ((I * invN) @ Ct)
            new_E = rng.binomial(S, -np.expm1(-foi * dt))
            new_I = rng.binomial(E, pI)
            new_R = rng.binomial(I, pR)

            S -= new_E
            E += new_E - new_I
            I += new_I - new_R
            R += new_R

            new_R_day += new_R
            new_I_day += new_I

        # deaths are reported Delta days after recovery
        tgt = (t - 1) + Delta                        # (N,) target day, per candidate
        inrange = tgt < T
        d_age = rng.binomial(new_R_day, ifr_row)     # (N, A)
        # (row, tgt) pairs are unique within a step (tgt is strictly increasing in t)
        deaths_daily[row[inrange], tgt[inrange]] = d_age.sum(axis=1)[inrange]
        if deaths_daily_age is not None:
            deaths_daily_age[row[inrange], tgt[inrange], :] = d_age[inrange]

        # FIX: the original stored `infections[:, t-1] = new_I`, i.e. only the
        # LAST sub-step of the day rather than the daily total. Deaths were never
        # affected (they correctly used the accumulated new_R_day), so the fit is
        # unchanged, but the reported incidence was ~1/daily_steps of the truth.
        if infec_daily is not None:
            infec_daily[:, t - 1] = new_I_day.sum(axis=1)
        if infec_daily_age is not None:
            infec_daily_age[:, t - 1, :] = new_I_day

        # S is a stock: store the level at the END of day t-1, summed over ages
        if S_daily is not None:
            S_daily[:, t - 1] = S.sum(axis=1)

    # The loop runs t = 1..T-1 and so simulates days 0..T-2; day T-1 is never
    # stepped. For the flows that is harmless (their last day is genuinely zero),
    # but a stock left at zero would look like total depletion, so carry the
    # final state forward explicitly.
    if S_daily is not None:
        S_daily[:, T - 1] = S.sum(axis=1)

    # ---- aggregate to weeks and drop the burn-in ----------------------------
    out = {"deaths_total": _to_weekly(deaths_daily, setup)}
    if deaths_daily_age is not None:
        out["deaths_by_age"] = _to_weekly(deaths_daily_age, setup)
    if infec_daily is not None:
        out["infections_total"] = _to_weekly(infec_daily, setup)
    if infec_daily_age is not None:
        out["infections_by_age"] = _to_weekly(infec_daily_age, setup)
    if S_daily is not None:
        out["S_total"] = _to_weekly_last(S_daily, setup)
    return out


def _to_weekly(daily: np.ndarray, setup: Setup) -> np.ndarray:
    """
    (N, T[, A]) daily -> (N, W[, A]) weekly, keeping only the bins from
    start_simu_date onwards. Equivalent to df.resample("W").sum()[cut_index:].

    For FLOWS (deaths, infections): each bin is the total over its days.
    """
    weekly = np.add.reduceat(daily, setup.week_starts, axis=1)
    return np.ascontiguousarray(weekly[:, setup.cut_index:], dtype=np.float32)


def _to_weekly_last(daily: np.ndarray, setup: Setup) -> np.ndarray:
    """
    (N, T) daily -> (N, W) weekly for a STOCK: the value on the day the bin
    closes (the week-ending Sunday), not a sum over the bin.

    Bin i covers days [week_starts[i], week_starts[i+1]), so it closes on
    week_starts[i+1] - 1; the last bin closes on the final simulated day. This is
    the same convention system B uses, where S is read off day 7t+6.
    """
    ends = np.append(setup.week_starts[1:], daily.shape[1]) - 1
    return np.ascontiguousarray(daily[:, ends][:, setup.cut_index:], dtype=np.float32)


def simulate_one(R0: float, inf_t0: float, Delta: int, setup: Setup,
                 seed: Optional[int] = None, **kwargs) -> Dict[str, np.ndarray]:
    """Convenience wrapper for a single candidate (used by validation/plots)."""
    rng = np.random.default_rng(seed)
    out = simulate_batch(np.array([R0]), np.array([inf_t0]), np.array([Delta]),
                         setup, rng, **kwargs)
    return {k: v[0] for k, v in out.items()}


def replicate(R0: float, inf_t0: float, Delta: int, setup: Setup, n_rep: int = 200,
              seed: Optional[int] = None, **kwargs) -> Dict[str, np.ndarray]:
    """
    Re-run one parameter set `n_rep` times to look at pure observation noise.

    NOTE ON REPRODUCIBILITY: candidates are simulated in shared-stream batches,
    so an individual pool trajectory cannot be re-derived from its index alone.
    Anything you may want later must be written to disk when the pool is built
    (see pool.store_* in london_abc.yaml). This function gives *fresh* draws from
    the same parameters, which is what you want for counterfactuals -- not for
    recovering a specific stored trajectory.
    """
    rng = np.random.default_rng(seed)
    return simulate_batch(np.full(n_rep, R0), np.full(n_rep, inf_t0),
                          np.full(n_rep, Delta, dtype=np.int64), setup, rng, **kwargs)
