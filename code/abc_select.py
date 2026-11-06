"""
ABC-with-budget selection: score the candidate pool, keep the best `n_accept`.

Two entry points, both operating on the SAME pre-simulated pool:

  run_retrospective()  one fit to the whole calibration period
  run_rolling()        one fit per rolling window x per n_history setting

WHAT A ROLLING WINDOW IS
------------------------
Let `t` index the weekly points (0 = 2020-03-15 ... W-1 = end_date) and let `t`
be the "current" week -- the last week whose data the forecaster is allowed to
see. For a window at `t` with history length `n`:

    calibration weeks : [t-n+1 .. t]        (n = "all" -> [0 .. t])
    forecast at h     : week t+h            h = 0 is the current week (nowcast)

    t-n+1        t-1   t    t+1  t+2  t+3  t+4
      |___________|____|     |    |    |    |
        scored on this     h=0  h=1  h=2  h=3  h=4
        (n points)         ^-- h=0 IS the last scored week

WHAT IS SAVED
-------------
Not trajectories -- INDICES. Each window stores the `n_accept` row indices of the
pool it accepted, so `pool["deaths_total"][idx]` is the full 25-week trajectory of
every accepted candidate, entire, from 2020-02-16 onwards. Any horizon (including
horizons you have not thought of yet) is a slice of that, with no re-simulation.

    <output>/selections/<tag>/retrospective_<error>.npz
    <output>/selections/<tag>/rolling_<error>_n<4|5|6|all>.npz
    <output>/selections/<tag>/rolling_meta.json

The rolling files are written in exactly the format system B uses in
<output>/selections_B/<tag>/ -- same file names, same npz keys, same meta schema --
so a single reader works on both systems. Each n_history carries its own error
(rolling.error_for_n), which is why the error is part of the file name and is also
stored inside the npz.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

import abc_priors
from abc_config import DEFAULT_CONFIG, Setup, _resolve, build_setup, load_config
from abc_errors import get_error
from abc_pool import open_pool

SCORE_CHUNK = 200_000     # rows scored at a time, keeps peak RAM bounded


# -----------------------------------------------------------------------------
# paths
# -----------------------------------------------------------------------------
def selection_dir(cfg: dict, tag: str) -> str:
    d = os.path.join(_resolve(cfg["paths"]["output"]), "selections", tag)
    os.makedirs(d, exist_ok=True)
    return d


# -----------------------------------------------------------------------------
# core primitives
# -----------------------------------------------------------------------------
def score_pool(sim: np.ndarray, obs: np.ndarray, weeks: slice,
               error_name: str, error_kwargs: Dict[str, Any]) -> np.ndarray:
    """
    Error of every pool member over `weeks`.

    sim : (N, W) or (N, W, A) array/memmap of simulated weekly deaths
    obs : (W,)   or (W, A)    observed weekly deaths
    """
    fn = get_error(error_name)
    obs_w = obs[weeks]
    N = sim.shape[0]
    err = np.empty(N, dtype=np.float64)
    for lo in range(0, N, SCORE_CHUNK):
        hi = min(lo + SCORE_CHUNK, N)
        err[lo:hi] = fn(obs_w, np.asarray(sim[lo:hi, weeks]), **error_kwargs)
    return err


def select_topk(err: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest errors, sorted best-first."""
    k = min(k, err.shape[0])
    part = np.argpartition(err, k - 1)[:k]
    return part[np.argsort(err[part])].astype(np.int32)


# -----------------------------------------------------------------------------
# window construction
# -----------------------------------------------------------------------------
def calibration_slice(t: int, n: Union[int, str],
                     first_week: int = 0) -> slice:
    """
    Weeks used to score a window whose current week is `t`.

    `first_week` is the earliest week allowed into the scoring slice. It defaults
    to 0, which is the behaviour every existing selection was made with, so no
    caller has to change. A run that scores from later than the anchor passes its
    own value: the London deaths are exactly zero in weeks 0..3, and a metric
    without a log shift cannot use those weeks at all.
    """
    if n == "all":
        return slice(first_week, t + 1)
    n = int(n)
    if n < 1:
        raise ValueError("n_history must be >= 1")
    return slice(max(first_week, t - n + 1), t + 1)


def window_grid(setup: Setup, rcfg: Dict[str, Any]) -> List[int]:
    """
    The `t` values to run.

    Windows are ALIGNED across n_history settings: the first window is late
    enough that even the longest finite history fits. Otherwise n=4 would get
    extra early windows that n=6 does not have, and the n-comparison would be
    confounded by which weeks each setting saw.
    """
    W = setup.W
    n_list = rcfg["n_history"]
    finite = [int(n) for n in n_list if n != "all"]
    need = max(finite + [int(rcfg.get("min_history", 1))]) if (finite or rcfg.get("min_history")) else 1

    first = rcfg.get("first_window")
    first = (need - 1) if first is None else int(first)

    last = rcfg.get("last_window")
    h_max = max(rcfg["horizons"])
    if last is None:
        last = W - 1 - h_max          # every horizon has observed data to score against
    elif last == "max":
        last = W - 1                  # forecast off the end of the data (unscoreable)
    else:
        last = int(last)

    step = int(rcfg.get("step", 1))
    if first > last:
        raise ValueError(
            f"no valid windows: first={first} > last={last} with W={W} weeks, "
            f"min history {need}, max horizon {h_max}. Widen simulation.end_date "
            f"or shrink rolling.n_history / rolling.horizons."
        )
    return list(range(first, last + 1, step))


# -----------------------------------------------------------------------------
# retrospective fit
# -----------------------------------------------------------------------------
def run_retrospective(config_path: str = DEFAULT_CONFIG, tag: str = "main",
                      error_name: Optional[str] = None,
                      error_kwargs: Optional[Dict[str, Any]] = None,
                      save: bool = True) -> Dict[str, np.ndarray]:
    cfg = load_config(config_path)
    setup = build_setup(cfg=cfg, config_path=config_path)
    pool = open_pool(cfg, tag)

    acfg, fcfg = cfg["abc"], cfg["retrospective"]
    error_name = error_name or fcfg.get("error", acfg["error"])
    error_kwargs = error_kwargs if error_kwargs is not None else fcfg.get("error_kwargs", {})
    k = int(acfg["n_accept"])

    sim, obs = _target_arrays(pool, setup, acfg["target"])
    weeks = slice(0, setup.W)

    err = score_pool(sim, obs, weeks, error_name, error_kwargs)
    idx = select_topk(err, k)
    params = np.asarray(pool["params"])[idx]

    print(f"[retrospective] error={error_name}{error_kwargs}  target={acfg['target']}  "
          f"N={sim.shape[0]:,} -> keep {len(idx):,}")
    print(f"  accepted error: best {err[idx[0]]:.4f}  worst {err[idx[-1]]:.4f}  "
          f"(pool median {np.median(err):.4f})")
    _report_posterior(params)

    out = {"idx": idx, "err": err[idx].astype(np.float32), "params": params,
           "weeks_lo": np.int32(0), "weeks_hi": np.int32(setup.W)}
    if save:
        path = os.path.join(selection_dir(cfg, tag), f"retrospective_{error_name}.npz")
        np.savez_compressed(path, all_err_quantiles=np.percentile(err, [0, 1, 5, 25, 50]), **out)
        print(f"  -> {path}")
    return out


# -----------------------------------------------------------------------------
# rolling forecasts
# -----------------------------------------------------------------------------
def run_rolling(config_path: str = DEFAULT_CONFIG, tag: str = "main",
                error_name: Optional[str] = None,
                error_kwargs: Optional[Dict[str, Any]] = None,
                n_history: Optional[Sequence[Union[int, str]]] = None,
                save: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    cfg = load_config(config_path)
    setup = build_setup(cfg=cfg, config_path=config_path)
    pool = open_pool(cfg, tag)

    acfg, rcfg = cfg["abc"], cfg["rolling"]
    n_list = list(n_history if n_history is not None else rcfg["n_history"])
    k = int(acfg["n_accept"])

    # Each n_history is scored with its own error, exactly as system B does.
    # An explicit --error on the command line overrides the map for every n.
    error_for_n, kwargs_for = _error_map(cfg, n_list, error_name, error_kwargs)

    sim, obs = _target_arrays(pool, setup, acfg["target"])
    all_params = np.asarray(pool["params"])
    ts = window_grid(setup, rcfg)

    print(f"[rolling] system A (age-structured SEIR, mobility "
          f"{cfg['model'].get('mobility', 'data')})  target={acfg['target']}  "
          f"N={sim.shape[0]:,}  keep {k:,}/window")
    print(f"  windows t = {ts[0]}..{ts[-1]} ({len(ts)} windows), "
          f"current week {setup.obs_weeks[ts[0]].date()} .. {setup.obs_weeks[ts[-1]].date()}")
    print(f"  horizons {rcfg['horizons']}  n_history {n_list}")
    print(f"  errors   " + ", ".join(f"n={n}:{error_for_n[str(n)]}" for n in n_list))

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for n in n_list:
        error_name = error_for_n[str(n)]
        error_kwargs = kwargs_for.get(error_name, {})
        nw = len(ts)
        idx = np.empty((nw, k), dtype=np.int32)
        errs = np.empty((nw, k), dtype=np.float32)
        params = np.empty((nw, k, all_params.shape[1]), dtype=np.float64)
        cal_lo = np.empty(nw, dtype=np.int32)

        for w, t in enumerate(ts):
            sl = calibration_slice(t, n)
            e = score_pool(sim, obs, sl, error_name, error_kwargs)
            sel = select_topk(e, k)
            idx[w], errs[w], params[w], cal_lo[w] = sel, e[sel], all_params[sel], sl.start

        res = {"window_t": np.asarray(ts, dtype=np.int32), "cal_lo": cal_lo,
               "idx": idx, "err": errs, "params": params}
        results[str(n)] = res

        acc = errs[:, -1]
        print(f"  n={str(n):>3}  ({error_name}{error_kwargs if error_kwargs else ''})")
        print(f"       acceptance threshold (worst kept error) "
              f"min {acc.min():.4f} / median {np.median(acc):.4f} / max {acc.max():.4f}")

        if save:
            # Same file name and same key set as system B's
            # selections_B/<tag>/rolling_<error>_n<n>.npz, so one reader works on
            # both. `error` and `param_names` live IN the npz, not only in the
            # meta, because the file is what gets shared.
            path = os.path.join(selection_dir(cfg, tag), f"rolling_{error_name}_n{n}.npz")
            np.savez_compressed(path, n_history=str(n), error=error_name,
                                # int64 explicitly: numpy's default integer is
                                # int32 on Windows and int64 on Linux, and the
                                # files must not differ by where they were made
                                horizons=np.asarray(rcfg["horizons"], dtype=np.int64),
                                param_names=np.array(abc_priors.PARAM_NAMES), **res)
            print(f"       -> {path}")

    if save:
        # rolling_meta.json, not rolling_<error>_meta.json: with one error per
        # n_history there is no single error to put in the file name. Schema
        # copied from system B's selections_B/<tag>/rolling_meta.json.
        meta = {"tag": tag, "system": f"A (age-structured SEIR, mobility "
                                      f"{cfg['model'].get('mobility', 'data')})",
                "n_accept": k, "n_history": [str(n) for n in n_list],
                "error_for_n": {str(n): error_for_n[str(n)] for n in n_list},
                "error_kwargs": kwargs_for,
                "target": acfg["target"],
                "horizons": list(rcfg["horizons"]), "window_t": ts,
                "obs_weeks": [str(d.date()) for d in setup.obs_weeks],
                "param_names": abc_priors.PARAM_NAMES}
        with open(os.path.join(selection_dir(cfg, tag), "rolling_meta.json"),
                  "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    return results


def _error_map(cfg: dict, n_list: Sequence[Union[int, str]],
               override: Optional[str], override_kwargs: Optional[Dict[str, Any]]):
    """
    {n_history: error name}, {error name: kwargs}.

    Read from rolling.error_for_n / rolling.error_kwargs_for, falling back to
    abc.error for any n that is not listed. `override` (the --error flag) forces
    one error for every n.
    """
    acfg, rcfg = cfg["abc"], cfg["rolling"]
    default = acfg["error"]
    default_kwargs = acfg.get("error_kwargs", {})

    if override is not None:
        kwargs = override_kwargs if override_kwargs is not None else default_kwargs
        return ({str(n): override for n in n_list}, {override: kwargs})

    table = {str(k): str(v) for k, v in (rcfg.get("error_for_n") or {}).items()}
    for_n = {str(n): table.get(str(n), default) for n in n_list}

    kwargs_for = dict(rcfg.get("error_kwargs_for") or {})
    for name in set(for_n.values()):
        if name not in kwargs_for:
            kwargs_for[name] = default_kwargs if name == default else {}
    return for_n, kwargs_for


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _target_arrays(pool: Dict[str, np.ndarray], setup: Setup, target: str):
    if target == "total":
        return pool["deaths_total"], setup.obs_total
    if target == "by_age":
        if "deaths_by_age" not in pool:
            raise KeyError("target='by_age' needs pool.store_deaths_by_age: true "
                           "in the yaml when the pool is built")
        return pool["deaths_by_age"], setup.obs_by_age
    raise ValueError(f"unknown abc.target {target!r} (use 'total' or 'by_age')")


def _report_posterior(params: np.ndarray) -> None:
    for j, name in enumerate(abc_priors.PARAM_NAMES):
        c = params[:, j]
        print(f"  {name:>6}: median {np.median(c):8.3f}   "
              f"90% CrI [{np.percentile(c, 5):.3f}, {np.percentile(c, 95):.3f}]")


def load_rolling(cfg: dict, tag: str, error_name: Optional[str], n: Union[int, str]):
    """
    The selection file for this n_history.

    `error_name=None` finds it whatever error it was scored with, which is what
    callers should pass now that each n_history has its own error. Give an
    explicit name only to disambiguate a folder holding several scorings of the
    same n.
    """
    d = selection_dir(cfg, tag)
    if error_name is not None:
        path = os.path.join(d, f"rolling_{error_name}_n{n}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return np.load(path, allow_pickle=False)

    found = sorted(glob.glob(os.path.join(d, f"rolling_*_n{n}.npz")))
    if not found:
        raise FileNotFoundError(os.path.join(d, f"rolling_*_n{n}.npz"))
    if len(found) > 1:
        raise ValueError(
            f"{len(found)} selection files for n={n} in {d} "
            f"({', '.join(os.path.basename(f) for f in found)}); pass error_name "
            f"to say which one, or delete the stale scorings."
        )
    return np.load(found[0], allow_pickle=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ABC-with-budget selection")
    ap.add_argument("mode", choices=["retro", "rolling", "both"], default="both", nargs="?")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--error", default=None, help="override the error function name")
    a = ap.parse_args()

    if a.mode in ("retro", "both"):
        run_retrospective(a.config, a.tag, a.error)
    if a.mode in ("rolling", "both"):
        run_rolling(a.config, a.tag, a.error)
