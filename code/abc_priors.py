"""
Prior sampling for the ABC-with-budget candidate pool.

The free parameters are the three of SEIR_model_general.SEIR() that are not
fixed epidemiological constants:

    R0      basic reproduction number
    inf_t0  number of individuals seeded at start_org_date
    Delta   delay in days between recovery and death being reported

Adding a parameter is a two-step change: add it to `priors:` in the yaml, then
thread it through abc_seir.simulate_batch(). Everything downstream (pool,
selection, forecasting) works off PARAM_NAMES and needs no edit.

`inf_t0` was called `i0` up to and including the main_0308 pool. It is the same
quantity; the name was changed so system A and system B label it identically and
one reader works on both (system B has always called it inf_t0). Pools built
before the rename still carry "i0" in their meta.json param_names.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

PARAM_NAMES: List[str] = ["R0", "inf_t0", "Delta"]


def sample_prior(cfg: Dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw `n` candidates from the priors.

    Returns
    -------
    (n, len(PARAM_NAMES)) float64 array, columns ordered as PARAM_NAMES.
    """
    priors = cfg["priors"]
    cols = [_draw(priors[name], n, rng) for name in PARAM_NAMES]
    return np.column_stack(cols)


def _draw(spec: Dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    dist = spec["dist"]
    lo, hi = float(spec["min"]), float(spec["max"])

    if dist == "uniform":
        return rng.uniform(lo, hi, n)
    if dist == "loguniform":
        if lo <= 0:
            raise ValueError("loguniform needs min > 0")
        return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    if dist == "randint":
        # inclusive on both ends, matching how Delta_min/Delta_max read
        return rng.integers(int(lo), int(hi) + 1, n).astype(np.float64)
    if dist == "normal":
        return rng.normal(spec["mean"], spec["sd"], n)
    raise ValueError(f"unknown prior distribution {dist!r}")


def as_dict(params: np.ndarray) -> Dict[str, np.ndarray]:
    """(n, P) array -> {'R0': (n,), 'inf_t0': (n,), 'Delta': (n,)} with Delta as int."""
    out = {name: params[:, j] for j, name in enumerate(PARAM_NAMES)}
    out["Delta"] = np.rint(out["Delta"]).astype(np.int64)
    return out


def prior_bounds(cfg: Dict[str, Any]) -> Dict[str, tuple]:
    """{name: (min, max)} -- handy for plotting posterior vs prior support."""
    return {name: (float(cfg["priors"][name]["min"]), float(cfg["priors"][name]["max"]))
            for name in PARAM_NAMES}
