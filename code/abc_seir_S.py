"""
Recover S(t) for the candidates that the SYSTEM B rolling selection accepted.

WHY THIS IS POSSIBLE AT ALL
---------------------------
abc_seir.py calls SEIR(..., seed=SEED + k), where k is the candidate's GLOBAL
row index in the 1e6 pool, and SEIR() starts with np.random.seed(int(seed)).
Every candidate's stochastic path is therefore a deterministic function of its
row index alone: re-running row k on its own reproduces exactly the trajectory
that is already stored in pools_B/<tag>/deaths_total.npy[k].

That claim is not taken on trust. Every re-run candidate's 25 weekly deaths are
compared element-for-element against the stored pool row, and the script aborts
if a single one differs -- because if the RNG stream has drifted (different
numpy, a threading backend instead of processes) then the S(t) recovered here
would belong to a DIFFERENT realisation than the deaths that were fitted, and
every Rt computed from it would be quietly wrong.

    => run this on the cluster, in the same venv that produced the pool.

WHAT IT DOES
------------
  1. read every ../output/selections_B/<tag>/rolling_*_n*.npz and take the UNION
     of their `idx`. 20 windows x 4 n_history x 1000 = 80,000 slots, but the same
     candidate is accepted by many windows, so the union is far smaller (~3% of
     the pool). Only the union is re-simulated.
  2. re-simulate those candidates, recording daily total S alongside the deaths.
  3. verify the deaths against the pool (see above).
  4. save S on the same fixed calendar grid the deaths use.

OUTPUT  ../output/S_B/<tag>/
    pool_idx.npy          (n_unique,)      int32, sorted -- rows of the pool
    S_weekly_total.npy    (n_unique, 25)   float32, S on each week-ending SUNDAY
    S_daily_total.npy     (n_unique, 175)  float32, S on every day of the grid
    meta.json

Rt then needs no simulation at all. For window w (current week t) and accepted
sample s:

    Rt[w, s] = R0[w, s] * S_weekly_total[row_of(idx[w, s]), t] / N

with R0 read straight out of the selection file's `params`. Use rows_for() to
turn pool indices into rows of these arrays.

    python abc_seir_S.py --tag seir_1M_0216 --limit 200   # smoke test, saves nothing
    python abc_seir_S.py --tag seir_1M_0216 --n-jobs 8    # the real thing
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from compartment_model_age_deaths_SEIR import SEIR
from constants import country_data, eps, mu

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(_HERE, "..", "data", "regions", "London"))
OUTPUT = os.path.normpath(os.path.join(_HERE, "..", "output"))

# ============================ SETTINGS =======================================
# These are COPIES of abc_seir.py's constants. abc_seir.py cannot be imported --
# it simulates at module level -- so they have to be duplicated, which means they
# can drift. _check_meta() below compares them against the meta.json that
# abc_seir_rolling.py wrote next to the pool, so a drift fails loudly instead of
# producing S(t) for the wrong trajectories.
SEED = 20200315
PRIORS = [
    ("R0",      "u",   1.3,  3.0),
    ("inf_t0",  "log", 1.0,  5000.0),
    ("Delta",   "int", 7,    35),
    ("delta_t", "int", 0,    150),
]
PARAM_NAMES = [p[0] for p in PRIORS]

ANCHOR = datetime(2020, 2, 16)        # first scored week (a Sunday)
SEED_REF = datetime(2020, 3, 8)       # start_org = SEED_REF - delta_t; NOT the anchor
END_DATE = datetime(2020, 8, 3)       # EXCLUSIVE, so the last simulated day is 08-02
EVAL_START = ANCHOR - timedelta(days=6)   # 2020-02-10, a Monday -> day index 0
OBS_FILE = "weekly_deaths_total.csv"

BASIN = "London"
CFG = country_data[BASIN]
HEMISPHERE = CFG["hemisphere"]
EPS, MU = eps, mu
REC_T0 = 0
SEASONALITY_MIN = 1.0
DETECTION_RATE = 1.0
DAILY_STEPS = 12
DEATHS_DELAY = "fixed"
# =============================================================================

Nk = pd.read_csv(f"{DATA}/demographic/pop.csv")["population"].values.astype(float)
ifr = np.loadtxt(f"{DATA}/epidemic/IFR_hetero.csv", delimiter=",")
C = np.loadtxt(f"{DATA}/contacts_matrix/contacts_all_locations.csv", delimiter=",")
NTOT = float(Nk.sum())

_obs = pd.read_csv(f"{DATA}/epidemic/{OBS_FILE}", parse_dates=["date"])
_obs = _obs[(_obs.date >= EVAL_START) & (_obs.date < END_DATE)]
WEEKS = _obs.date.values
NW = len(WEEKS)                       # 25
NDAYS = 7 * NW                        # 175
if NW == 0 or pd.Timestamp(WEEKS[0]) != pd.Timestamp(ANCHOR):
    raise SystemExit(f"{OBS_FILE} has no row for ANCHOR {ANCHOR.date()}")


def pool_dir(tag: str) -> str:
    return os.path.join(OUTPUT, "pools_B", tag)


def selection_dir(tag: str) -> str:
    return os.path.join(OUTPUT, "selections_B", tag)


def s_dir(tag: str) -> str:
    return os.path.join(OUTPUT, "S_B", tag)


# -----------------------------------------------------------------------------
# guards
# -----------------------------------------------------------------------------
def _check_meta(tag: str) -> None:
    """The settings above must match the ones the pool was built with."""
    path = os.path.join(pool_dir(tag), "meta.json")
    if not os.path.exists(path):
        print(f"  ! {path} not found -- the constants in this file cannot be "
              f"cross-checked against the pool. Proceeding on the assumption that "
              f"they still match abc_seir.py.")
        return
    with open(path, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    want = {"seed": SEED, "anchor": str(ANCHOR.date()),
            "seed_ref": str(SEED_REF.date()), "W": NW}
    for key, val in want.items():
        if key in m and m[key] != val:
            raise SystemExit(
                f"{path} says {key}={m[key]!r} but this script assumes {val!r}. "
                f"The constants have drifted from the run that produced the pool; "
                f"the recovered S(t) would belong to different trajectories.")
    if "priors" in m:
        mine = [[p[0], p[1], float(p[2]), float(p[3])] for p in PRIORS]
        theirs = [[p[0], p[1], float(p[2]), float(p[3])] for p in m["priors"]]
        if mine != theirs:
            raise SystemExit(f"{path} has priors {theirs}, this script has {mine}")
    print(f"  settings cross-checked against {os.path.relpath(path, OUTPUT)}: OK")


def draw_params(n: int) -> np.ndarray:
    """Redraw the 1e6 prior draws exactly as abc_seir.py did."""
    rng = np.random.default_rng(SEED)
    cols = []
    for name, kind, lo, hi in PRIORS:
        if kind == "u":
            v = rng.uniform(lo, hi, n)
        elif kind == "log":
            v = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        elif kind == "log10":
            v = 10.0 ** rng.uniform(lo, hi, n)
        elif kind == "int":
            v = rng.integers(int(lo), int(hi) + 1, n).astype(float)
        elif kind == "fixed":
            v = np.full(n, float(lo))
        else:
            raise ValueError(f"unknown prior kind {kind!r} for {name}")
        cols.append(v)
    return np.column_stack(cols)


def accepted_union(tag: str) -> np.ndarray:
    """Sorted union of the pool rows accepted by any window of any n_history."""
    files = sorted(glob.glob(os.path.join(selection_dir(tag), "rolling_*_n*.npz")))
    if not files:
        raise SystemExit(f"no rolling_*.npz in {selection_dir(tag)}; "
                         f"run abc_seir_rolling.py first")
    slots = 0
    parts = []
    for f in files:
        z = np.load(f)
        idx = np.asarray(z["idx"])
        slots += idx.size
        parts.append(idx.ravel())
        print(f"  {os.path.basename(f):<32} {idx.shape[0]:>3} windows x "
              f"{idx.shape[1]:,} accepted")
    union = np.unique(np.concatenate(parts)).astype(np.int32)
    print(f"  {slots:,} accepted slots -> {len(union):,} unique candidates "
          f"({len(union)/slots:.1%})")
    return union


# -----------------------------------------------------------------------------
# the re-run
# -----------------------------------------------------------------------------
def run_one(k: int, row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-simulate pool row k and return (25 weekly deaths, 175 daily total S).

    This is abc_seir.run() with S carried along. The mapping onto the fixed
    calendar grid is IDENTICAL and must stay that way: deaths before start_org
    are zero because the epidemic had not been seeded, and S before start_org is
    the whole population for the same reason.
    """
    R0, inf_t0, Delta, delta_t = row
    start_org = SEED_REF - timedelta(days=int(delta_t))
    T_k = (END_DATE - start_org).days
    dates_k = [start_org + timedelta(days=d) for d in range(T_k)]

    out = SEIR(inf_t0=inf_t0, rec_t0=REC_T0, Nk=Nk, T=T_k, R0=R0,
               eps=EPS, mu=MU,
               ifr=ifr, Delta=int(Delta), C=C, detection_rate=DETECTION_RATE,
               dates=dates_k, hemisphere=HEMISPHERE, seasonality_min=SEASONALITY_MIN,
               deaths_delay=DEATHS_DELAY, daily_steps=DAILY_STEPS, seed=SEED + int(k))

    off = (start_org - EVAL_START).days      # >0 seeded after the grid opens

    d = np.asarray(out["deaths"], dtype=float)
    d_grid = np.concatenate([np.zeros(off), d]) if off >= 0 else d[-off:]
    weekly = d_grid.reshape(NW, 7).sum(axis=1)

    s = np.asarray(out["S"], dtype=float)    # compartments[0].sum(axis=0), daily
    s_grid = np.concatenate([np.full(off, NTOT), s]) if off >= 0 else s[-off:]

    return weekly, s_grid.astype(np.float32)


def main(tag: str, n_jobs: int, limit: int) -> None:
    print(f"[S recovery] system B, pool '{tag}'")
    _check_meta(tag)

    sim_path = os.path.join(pool_dir(tag), "deaths_total.npy")
    if not os.path.exists(sim_path):
        raise SystemExit(f"{sim_path} not found -- the verification step needs the "
                         f"pool. Build it with abc_seir_rolling.py --pool-only.")
    pool = np.load(sim_path, mmap_mode="r")
    N, W = pool.shape
    if W != NW:
        raise SystemExit(f"pool has {W} weeks, this script assumes {NW}")

    print(f"\nselections:")
    union = accepted_union(tag)
    if limit:
        union = union[:limit]
        print(f"  --limit {limit}: only the first {len(union):,} will be run, "
              f"and NOTHING will be saved")

    params = draw_params(N)[union]

    print(f"\nre-simulating {len(union):,} candidates on {n_jobs} cpus "
          f"(each is a full run from its own start_org to {(END_DATE - timedelta(days=1)).date()})",
          flush=True)
    t0 = time.time()
    # loky = separate PROCESSES. This is not a preference: SEIR() seeds the global
    # numpy RandomState, so a threading backend would let the workers trample each
    # other's stream and the verification below would fail.
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(run_one)(int(k), params[i]) for i, k in enumerate(union))
    el = time.time() - t0
    print(f"\ndone in {el/60:.1f} min ({el/len(union)*1e3:.0f} ms/candidate wall, "
          f"{el*n_jobs/len(union):.2f} s/candidate cpu)")

    deaths = np.array([r[0] for r in res], dtype=np.float64)
    s_daily = np.array([r[1] for r in res], dtype=np.float32)

    # --- the gate ------------------------------------------------------------
    stored = np.asarray(pool[union], dtype=np.float32)
    same = deaths.astype(np.float32) == stored
    n_bad = int((~same.all(axis=1)).sum())
    if n_bad:
        w = np.where(~same.all(axis=1))[0][:5]
        raise SystemExit(
            f"\nVERIFICATION FAILED: {n_bad:,} of {len(union):,} re-simulated "
            f"candidates do not reproduce the stored pool row.\n"
            f"  first offenders (pool rows): {union[w].tolist()}\n"
            f"  The per-candidate RNG stream has drifted -- most likely a different\n"
            f"  numpy version, or joblib running threads instead of processes. The\n"
            f"  S(t) recovered here would belong to a DIFFERENT realisation than the\n"
            f"  deaths that were fitted, so it is not usable. Do not save it.")
    print(f"verification: all {len(union):,} candidates reproduce their stored "
          f"pool row exactly -- S(t) belongs to the fitted trajectories")

    # --- weekly S, at each week-ending SUNDAY --------------------------------
    # day index 7j+6 is the Sunday that closes week j (grid day 0 = 2020-02-10,
    # a Monday). This is the day the rolling windows are labelled with.
    s_weekly = np.ascontiguousarray(s_daily[:, 6::7])
    assert s_weekly.shape[1] == NW

    frac = s_weekly / NTOT
    print(f"\nS/N over the {NW} weeks, across the {len(union):,} candidates:")
    print(f"  week 0  ({pd.Timestamp(WEEKS[0]).date()}):  median {np.median(frac[:, 0]):.4f}"
          f"   5-95% [{np.percentile(frac[:, 0], 5):.4f}, {np.percentile(frac[:, 0], 95):.4f}]")
    print(f"  week {NW-1} ({pd.Timestamp(WEEKS[-1]).date()}):  median {np.median(frac[:, -1]):.4f}"
          f"   5-95% [{np.percentile(frac[:, -1], 5):.4f}, {np.percentile(frac[:, -1], 95):.4f}]")
    print(f"  => Rt = R0 * S/N is at most {(1 - np.median(frac[:, -1]))*100:.1f}% "
          f"below R0 by the end, for the median candidate")

    if limit:
        print(f"\n--limit was set, so nothing was written. Drop it to save.")
        return

    d = s_dir(tag)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "pool_idx.npy"), union)
    np.save(os.path.join(d, "S_weekly_total.npy"), s_weekly)
    np.save(os.path.join(d, "S_daily_total.npy"), s_daily)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "system": "B (SEIR null, no mobility)",
                   "n_unique": int(len(union)), "pool_n": int(N),
                   "N_total": NTOT,
                   "seed": SEED, "anchor": str(ANCHOR.date()),
                   "seed_ref": str(SEED_REF.date()),
                   "grid_day0": str(EVAL_START.date()), "n_days": NDAYS,
                   "weeks": [str(pd.Timestamp(x).date()) for x in WEEKS],
                   "weekly_day_index": [7 * j + 6 for j in range(NW)],
                   "verified_against_pool": True,
                   "param_names": PARAM_NAMES}, fh, indent=2)

    print(f"\n-> {os.path.join(d, 'pool_idx.npy')}        {union.shape}")
    print(f"-> {os.path.join(d, 'S_weekly_total.npy')}  {s_weekly.shape}  "
          f"({s_weekly.nbytes/1e6:.1f} MB)")
    print(f"-> {os.path.join(d, 'S_daily_total.npy')}   {s_daily.shape}  "
          f"({s_daily.nbytes/1e6:.1f} MB)")


# -----------------------------------------------------------------------------
# helper for the plotting script
# -----------------------------------------------------------------------------
def rows_for(pool_idx: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """
    Map pool row numbers (what the selections store) to rows of S_*_total.npy.

    pool_idx is sorted, so this is a searchsorted. Raises if anything asked for
    was not re-simulated, rather than silently returning row 0.
    """
    pos = np.searchsorted(pool_idx, wanted)
    pos = np.clip(pos, 0, len(pool_idx) - 1)
    if not np.array_equal(pool_idx[pos], wanted):
        missing = np.unique(np.asarray(wanted)[pool_idx[pos] != wanted])
        raise SystemExit(f"{len(missing):,} pool rows are not in S_B (e.g. "
                         f"{missing[:5].tolist()}) -- the selections have changed "
                         f"since abc_seir_S.py ran. Re-run it.")
    return pos


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="recover S(t) for accepted candidates")
    ap.add_argument("--tag", default="seir_1M_0216")
    ap.add_argument("--n-jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N of the union and save nothing "
                         "(smoke test: checks reproducibility cheaply)")
    a = ap.parse_args()
    main(a.tag, a.n_jobs, a.limit)
