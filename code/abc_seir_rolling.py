"""
Rolling-window selection for SYSTEM B (SEIR null model, no mobility).

This is the System-B counterpart of `abc_select.py rolling`. It does NOT simulate
anything -- `abc_seir.py` already produced 1e6 full 22-week trajectories, split
across ../output/parts/<tag>/part*.npz. Every rolling window is just a
re-scoring of that same pool.

Two steps, both driven from here:

  1. build_pool()   merge the 200 parts once into a single memmap-able pool
                        ../output/pools_B/<tag>/deaths_total.npy  (N, 22) float32
                        ../output/pools_B/<tag>/params.npy        (N,  4) float64
                        ../output/pools_B/<tag>/meta.json
                    Kept separate from System A's ../output/pools/ on purpose --
                    the two systems have different models, different eps, and
                    different parameter vectors, and their products must not be
                    mistaken for each other.

  2. run_rolling()  for each n_history and each window, score the whole pool over
                    that window's calibration weeks and keep the best n_accept.
                    Saves INDICES and PARAMETERS, not trajectories:

                        ../output/selections_B/<tag>/rolling_<error>_n<n>.npz
                            window_t (n_win,)                which week is "now"
                            cal_lo   (n_win,)                first scored week
                            idx      (n_win, n_accept)       rows of the pool
                            err      (n_win, n_accept)       their errors
                            params   (n_win, n_accept, 4)    R0, inf_t0, Delta, delta_t

                    params[w] IS the posterior sample for window w -- each window
                    accepts a different set of candidates, which is the point.

WHICH ERROR
-----------
n_history 4, 5, 6 are scored with plain `wmape`; n_history "all" with
`biased_wmape` (recency-weighted mean |log ratio|). See ERROR_FOR_N below and the
note there about what that does to the comparison between them.

The window geometry, horizons and n_accept are read from london_abc.yaml so they
are guaranteed identical to System A's, making the two directly comparable
window-for-window. Only the `rolling:` and `abc.n_accept` keys are read; none of
System A's model machinery (contact matrices, mobility, IFR) is touched.

    python abc_seir_rolling.py --tag seir_1M_0308
    python abc_seir_rolling.py --tag seir_1M_0308 --rebuild-pool
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

# Pure algorithm primitives, shared with System A so the two run the SAME
# selection logic. These touch nothing but arrays -- no config, no model.
from abc_select import calibration_slice, score_pool, select_topk, window_grid

# Resolved against THIS FILE, not the working directory, so the scripts behave
# the same whether they are run from code/, from an IDE with some other working
# directory, or from a SLURM job.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(_HERE, "..", "data", "regions", "London"))
OUTPUT = os.path.normpath(os.path.join(_HERE, "..", "output"))
CONFIG = os.path.join(_HERE, "london_abc.yaml")

# Must match abc_seir.py. The parts store only `sim`; the parameters are
# regenerated from the seed, so if this list drifts from abc_seir.py the
# parameters get attached to the wrong candidates. _load_params() cross-checks
# against the copy merge_abc_seir.sh wrote, which is the real guard.
SEED = 20200315
PARAM_NAMES = ["R0", "inf_t0", "Delta", "delta_t"]
PRIORS = [
    ("R0",      "u",   1.3,  3.0),
    ("inf_t0",  "log", 1.0,  5000.0),
    ("Delta",   "int", 7,    35),
    ("delta_t", "int", 0,    150),
]

ANCHOR = "2020-02-16"          # first scored week; must match abc_seir.py
END_WEEK = "2020-08-02"        # last scored week (inclusive) -> 25 weeks
SEED_REF = "2020-03-08"        # start_org = SEED_REF - delta_t; NOT the anchor
OBS_FILE = "weekly_deaths_total.csv"

# Seeding was decoupled from the anchor when the window moved to 2020-02-16. The
# candidates are otherwise unchanged, so the new 25-week pool must agree with the
# old 22-week one on its last 22 columns. build_pool() checks this if it finds
# the old pool -- see _verify_against_previous().
PREV_POOL_TAG = "seir_1M_0308"
PREV_OFFSET = 3                # new pool has 3 extra weeks at the front

# n=4/5/6 use plain wMAPE, n="all" uses the paper's biased wMAPE.
#
# These are NOT on the same scale: wMAPE is sum|y-yhat|/sum(y), biased wMAPE is a
# weighted mean of |log(y/yhat)|. Error VALUES and acceptance thresholds cannot
# be compared across the two. Forecast skill (WIS, coverage, MAE) is computed
# from the forecasts themselves and stays comparable.
ERROR_FOR_N = {"4": "wmape", "5": "wmape", "6": "wmape", "all": "biased_wmape"}
ERROR_KWARGS = {"wmape": {}, "biased_wmape": {"shift": 1.0}}


# -----------------------------------------------------------------------------
# paths
# -----------------------------------------------------------------------------
def pool_dir(tag: str) -> str:
    return os.path.join(OUTPUT, "pools_B", tag)


def selection_dir(tag: str) -> str:
    d = os.path.join(OUTPUT, "selections_B", tag)
    os.makedirs(d, exist_ok=True)
    return d


def parts_dir(tag: str) -> str:
    return os.path.join(OUTPUT, "parts", tag)


# -----------------------------------------------------------------------------
# observations
# -----------------------------------------------------------------------------
def load_obs() -> Tuple[np.ndarray, np.ndarray]:
    """The 22 scored weeks, identical to what abc_seir.py fitted against."""
    obs = pd.read_csv(f"{DATA}/epidemic/{OBS_FILE}", parse_dates=["date"])
    obs = obs[(obs.date >= ANCHOR) & (obs.date <= END_WEEK)]
    weeks = obs.date.values
    y = obs.total.values.astype(np.float64)
    if len(y) == 0 or str(pd.Timestamp(weeks[0]).date()) != ANCHOR:
        raise SystemExit(f"{OBS_FILE} does not start at {ANCHOR}")
    return weeks, y


# -----------------------------------------------------------------------------
# parameters
# -----------------------------------------------------------------------------
def draw_params(n: int) -> np.ndarray:
    """
    Redraw the priors exactly as abc_seir.py did. Deterministic given SEED and
    PRIORS, which is how every array task and the merge agreed on them.
    """
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


def _verify_params(params: np.ndarray, tag: str) -> None:
    """
    Cross-check the redrawn parameters against the copy merge_abc_seir.sh saved.
    That file was written by the same process that produced the trajectories, so
    agreeing with it proves PRIORS here still matches abc_seir.py.
    """
    path = os.path.join(OUTPUT, f"{tag}_all.npz")
    if not os.path.exists(path):
        print(f"  ! {os.path.basename(path)} not found -- cannot cross-check the "
              f"parameters. They come from SEED={SEED} and the PRIORS list in this "
              f"file; make sure it still matches abc_seir.py.")
        return
    z = np.load(path)
    for j, name in enumerate(PARAM_NAMES):
        if name not in z:
            raise SystemExit(f"{path} has no '{name}'; PRIORS has changed since the run")
        # that copy is float32, so compare at float32 tolerance
        if not np.allclose(params[:, j], z[name].astype(np.float64), rtol=1e-5, atol=1e-5):
            raise SystemExit(
                f"redrawn '{name}' does not match {os.path.basename(path)}. The PRIORS "
                f"list or SEED in {__file__} has drifted from abc_seir.py -- the "
                f"parameters would be attached to the wrong candidates.")
    print(f"  parameters cross-checked against {os.path.basename(path)}: OK")


# -----------------------------------------------------------------------------
# step 1: build the pool
# -----------------------------------------------------------------------------
def _own_priors() -> list:
    """PRIORS as plain JSON-comparable data, matching what abc_seir.py records."""
    return json.loads(json.dumps([list(p) for p in PRIORS]))


def _part_priors(z) -> Optional[list]:
    """The priors a part npz was drawn from, or None if it predates the field."""
    return json.loads(str(z["priors"])) if "priors" in z else None


def _verify_against_previous(sim: np.ndarray, tag: str) -> None:
    """
    Decoupling seeding from the anchor means this pool's trajectories should be
    IDENTICAL to the previous one -- same start_org, same T, same per-candidate
    seed -- just with PREV_OFFSET more weeks kept at the front. That makes the
    re-run checkable instead of merely believed.

    That invariant only holds while the PRIORS are the ones PREV_POOL_TAG was
    drawn from. Change any of them and every candidate changes, so the
    trajectories SHOULD differ and this check would abort a perfectly good pool.
    The shapes alone do not catch it: a 1e6 x 25 pool built from different priors
    still lines up with the 1e6 x 22 previous one.
    """
    prev = os.path.join(pool_dir(PREV_POOL_TAG), "deaths_total.npy")
    if tag == PREV_POOL_TAG or not os.path.exists(prev):
        return
    prev_meta = os.path.join(pool_dir(PREV_POOL_TAG), "meta.json")
    if os.path.exists(prev_meta):
        with open(prev_meta, encoding="utf-8") as fh:
            prev_priors = json.load(fh).get("priors")
        if prev_priors is not None and prev_priors != _own_priors():
            changed = [a[0] for a, b in zip(_own_priors(), prev_priors) if a != b]
            print(f"  priors differ from '{PREV_POOL_TAG}' on {changed} -- every "
                  f"candidate differs by design, skipping the consistency check")
            return
    old = np.load(prev, mmap_mode="r")
    if old.shape[0] != sim.shape[0] or old.shape[1] + PREV_OFFSET != sim.shape[1]:
        print(f"  ! {PREV_POOL_TAG} is {old.shape}, not comparable with {sim.shape} "
              f"-- skipping the consistency check")
        return
    # compare in blocks; the whole thing is only 88 MB but the memmap is on disk
    bad = 0
    for lo in range(0, sim.shape[0], 200_000):
        hi = min(lo + 200_000, sim.shape[0])
        if not np.array_equal(sim[lo:hi, PREV_OFFSET:], np.asarray(old[lo:hi])):
            bad += 1
    if bad:
        raise SystemExit(
            f"CONSISTENCY CHECK FAILED: this pool disagrees with {PREV_POOL_TAG} on "
            f"the {old.shape[1]} weeks they share. Seeding was supposed to be "
            f"unchanged (start_org = {SEED_REF} - delta_t), so the trajectories "
            f"should be identical. Something else changed -- do not use this pool.")
    print(f"  consistency: last {old.shape[1]} weeks match pool '{PREV_POOL_TAG}' "
          f"exactly -- seeding was unaffected by the window move")


def build_pool(tag: str, rebuild: bool = False) -> str:
    d = pool_dir(tag)
    sim_path = os.path.join(d, "deaths_total.npy")
    if os.path.exists(sim_path) and not rebuild:
        print(f"pool already at {d} (use --rebuild-pool to redo it)")
        return d

    files = sorted(glob.glob(os.path.join(parts_dir(tag), "part*.npz")))
    if not files:
        raise SystemExit(f"no parts in {parts_dir(tag)}; run job_abc_seir.sh first")

    # first part tells us the shape and the settings everything must agree with
    z0 = np.load(files[0])
    N, W, seed = int(z0["n"]), int(z0["nw"]), int(z0["seed"])
    dt = (int(z0["dt_min"]), int(z0["dt_max"]))
    if seed != SEED:
        raise SystemExit(f"parts were simulated with seed {seed}, this script assumes {SEED}")
    if dt != (int(PRIORS[3][2]), int(PRIORS[3][3])):
        raise SystemExit(f"parts have delta_t in {list(dt)}, PRIORS here says "
                         f"[{PRIORS[3][2]}, {PRIORS[3][3]}]")
    if "seed_ref" not in z0:
        raise SystemExit(f"{files[0]} predates the seeding/anchor split: it used "
                         f"start_org = ANCHOR - delta_t, so every trajectory is "
                         f"shifted relative to this script's assumptions. Re-simulate.")
    if str(z0["seed_ref"]) != SEED_REF:
        raise SystemExit(f"parts were seeded relative to {str(z0['seed_ref'])}, "
                         f"this script assumes {SEED_REF}")
    # The delta_t check above covers one prior; this covers all four. Parts drawn
    # from inf_t0 in [1, 1e4] and from [1, 5000] agree on n, seed, nw, dt_min and
    # dt_max, so before `priors` was recorded nothing here could tell them apart.
    # Parts written by an older abc_seir.py do not carry the field; they are the
    # 1e4 runs, and there is no way to confirm that from the file, so say so
    # rather than pretend the check ran.
    priors = _part_priors(z0)
    if priors is None:
        print(f"  ! {os.path.basename(files[0])} predates the recorded-priors field. "
              f"PRIORS cannot be verified against it -- it is assumed to match the "
              f"list in this file. Re-simulate if that is in doubt.")
    elif priors != _own_priors():
        raise SystemExit(
            f"parts were drawn from priors\n    {priors}\n"
            f"but PRIORS in {os.path.basename(__file__)} says\n    {_own_priors()}\n"
            f"The parameters would be attached to the wrong candidates. Point --tag "
            f"at the pool these parts belong to, or restore the matching PRIORS.")

    print(f"building pool '{tag}': {len(files)} parts, N={N:,}, W={W}")
    sim = np.empty((N, W), dtype=np.float32)
    seen = np.zeros(N, dtype=bool)
    for f in files:
        z = np.load(f)
        if (int(z["n"]), int(z["seed"]), int(z["nw"])) != (N, seed, W):
            raise SystemExit(f"{f} disagrees with {files[0]} on n/seed/nw")
        if (int(z["dt_min"]), int(z["dt_max"])) != dt:
            raise SystemExit(f"{f} has a different delta_t range from {files[0]}")
        if _part_priors(z) != priors:
            raise SystemExit(f"{f} was drawn from different priors than "
                             f"{files[0]} -- two runs have been mixed in "
                             f"{parts_dir(tag)}")
        a, b = int(z["lo"]), int(z["hi"])
        sim[a:b], seen[a:b] = z["sim"], True
    if not seen.all():
        raise SystemExit(f"{(~seen).sum():,} candidates are missing from the parts")

    params = draw_params(N)
    _verify_params(params, tag)
    _verify_against_previous(sim, tag)

    os.makedirs(d, exist_ok=True)
    np.save(sim_path, sim)
    np.save(os.path.join(d, "params.npy"), params)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "system": "B (SEIR null, no mobility)",
                   "n": N, "W": W, "seed": seed,
                   "anchor": ANCHOR, "end_week": END_WEEK, "seed_ref": SEED_REF,
                   "obs_file": OBS_FILE,
                   "param_names": PARAM_NAMES,
                   "priors": [[p[0], p[1], p[2], p[3]] for p in PRIORS],
                   "n_parts": len(files),
                   "source": os.path.relpath(parts_dir(tag), OUTPUT)}, fh, indent=2)

    print(f"  -> {sim_path}  {sim.shape} {sim.dtype}  "
          f"({os.path.getsize(sim_path)/1e6:.0f} MB)")
    print(f"  -> {os.path.join(d, 'params.npy')}  {params.shape}")
    return d


def open_pool(tag: str) -> Dict[str, np.ndarray]:
    """
    The selections store INDICES, so anything that needs actual trajectories --
    plotting, forecast scoring -- needs this file. It is built where the parts
    are, i.e. usually on the cluster; copy it down if you want to work locally.
    """
    d = pool_dir(tag)
    sim_path = os.path.join(d, "deaths_total.npy")
    if not os.path.exists(sim_path):
        raise SystemExit(
            f"{sim_path} not found.\n"
            f"  The rolling selections only store row indices into this pool, so it is\n"
            f"  needed to turn them back into trajectories. Either:\n"
            f"    - build it here:   python abc_seir_rolling.py --tag {tag} --pool-only\n"
            f"                       (needs ../output/parts/{tag}/)\n"
            f"    - or copy it from wherever the parts were simulated:\n"
            f"        scp -r <host>:<proj>/output/pools_B/{tag} ../output/pools_B/")
    pool = {"deaths_total": np.load(sim_path, mmap_mode="r")}
    # only the selections' own copy of the parameters is needed for plotting, so
    # a pool without params.npy is still usable
    par = os.path.join(d, "params.npy")
    if os.path.exists(par):
        pool["params"] = np.load(par, mmap_mode="r")
    return pool


# -----------------------------------------------------------------------------
# step 2: rolling selection
# -----------------------------------------------------------------------------
def _rolling_config() -> Tuple[dict, int]:
    """The `rolling:` block and n_accept, read from System A's yaml verbatim."""
    with open(CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg["rolling"], int(cfg["abc"]["n_accept"])


def run_rolling(tag: str, save: bool = True,
                only_n: Optional[List[str]] = None,
                error_for_all: Optional[str] = None,
                out_tag: Optional[str] = None,
                ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Score the pool window by window and keep the best n_accept of each.

    The three overrides exist so one alternative selection can be made without
    touching the defaults, which several other files depend on:

      only_n         restrict to some of the yaml n_history values, so a run for
                     n="all" does not rewrite the n=4/5/6 files as a side effect.
      error_for_all  replace the distance used for n_history="all". ERROR_FOR_N
                     itself must NOT be edited: job_abc_seir_smc.sh asserts it is
                     identical to the copy in abc_seir_smc.py and aborts if not.
      out_tag        write the selection somewhere other than the pool tag, so
                     selections_B/<pool tag>/ and its rolling_meta.json are left
                     alone. The pool still comes from `tag`.

    A distance whose name ends in _nz drops the weeks with no observed deaths.
    A window whose whole calibration slice is zero therefore has nothing to score
    and is skipped -- for London that is t=3, weeks 0..3 (2020-02-16..03-08).
    """
    weeks, y = load_obs()
    pool = open_pool(tag)
    if "params" not in pool:
        raise SystemExit(f"{os.path.join(pool_dir(tag), 'params.npy')} is missing. "
                         f"Plotting does not need it but selection does -- copy the "
                         f"whole pool directory, or rebuild with --pool-only.")
    sim, all_params = pool["deaths_total"], np.asarray(pool["params"])
    N, W = sim.shape
    if W != len(y):
        raise SystemExit(f"pool has {W} weeks but the observations have {len(y)}")

    rcfg, k = _rolling_config()
    n_list = list(rcfg["n_history"])
    if only_n is not None:
        want = [str(x) for x in only_n]
        n_list = [n for n in n_list if str(n) in want]
        if not n_list:
            raise SystemExit("--n-history %s matches none of %s"
                             % (want, [str(n) for n in rcfg["n_history"]]))

    def error_for(key: str) -> str:
        if key == "all" and error_for_all is not None:
            return error_for_all
        return ERROR_FOR_N[key]

    out_dir = selection_dir(out_tag if out_tag else tag)

    class _S:                                  # window_grid() only reads .W
        pass
    s = _S()
    s.W = W
    ts = window_grid(s, rcfg)

    print(f"[rolling] system B, pool '{tag}': N={N:,}  W={W}  keep {k:,}/window")
    print(f"  weeks     {pd.Timestamp(weeks[0]).date()} .. {pd.Timestamp(weeks[-1]).date()}"
          f"   observed total {y.sum():,.0f}")
    print(f"  windows   t = {ts[0]}..{ts[-1]} ({len(ts)}), current week "
          f"{pd.Timestamp(weeks[ts[0]]).date()} .. {pd.Timestamp(weeks[ts[-1]]).date()}")
    print(f"  horizons  {rcfg['horizons']}   n_history {n_list}")
    print(f"  errors    " + ", ".join(f"n={n}:{error_for(str(n))}" for n in n_list))
    print(f"  writing   {out_dir}")

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for n in n_list:
        key = str(n)
        error_name = error_for(key)
        kw = ERROR_KWARGS.get(error_name, {})

        # _nz distances throw the zero weeks away, so a window whose slice holds
        # no deaths at all cannot be scored. Drop those windows rather than let
        # the distance raise halfway through.
        ts_n = list(ts)
        if error_name.endswith("_nz"):
            ts_n = [t for t in ts if (y[calibration_slice(t, n)] > 0).any()]
            if len(ts_n) < len(ts):
                dropped = [t for t in ts if t not in ts_n]
                print(f"  n={key}: {error_name} drops windows {dropped} "
                      f"(no observed deaths in their calibration weeks)")

        nw = len(ts_n)
        idx = np.empty((nw, k), dtype=np.int32)
        errs = np.empty((nw, k), dtype=np.float32)
        params = np.empty((nw, k, all_params.shape[1]), dtype=np.float64)
        cal_lo = np.empty(nw, dtype=np.int32)

        for w, t in enumerate(ts_n):
            sl = calibration_slice(t, n)
            e = score_pool(sim, y, sl, error_name, kw)
            sel = select_topk(e, k)
            idx[w], errs[w], params[w], cal_lo[w] = sel, e[sel], all_params[sel], sl.start

        res = {"window_t": np.asarray(ts_n, dtype=np.int32), "cal_lo": cal_lo,
               "idx": idx, "err": errs, "params": params}
        results[key] = res

        acc = errs[:, -1]
        print(f"\n  n={key:>3}  ({error_name}{kw if kw else ''})")
        print(f"       acceptance threshold: min {acc.min():.4f} / "
              f"median {np.median(acc):.4f} / max {acc.max():.4f}")
        for j, name in enumerate(PARAM_NAMES):
            med = np.median(params[:, :, j], axis=1)
            print(f"       {name:>8} per-window median: "
                  f"{med.min():.3g} .. {med.max():.3g}")

        if save:
            p = os.path.join(out_dir, f"rolling_{error_name}_n{key}.npz")
            np.savez_compressed(p, n_history=key, error=error_name,
                                horizons=np.asarray(rcfg["horizons"]),
                                param_names=np.array(PARAM_NAMES), **res)
            print(f"       -> {p}")

    if save:
        meta = {"tag": tag, "system": "B (SEIR null, no mobility)",
                "n_accept": k, "n_history": [str(n) for n in n_list],
                "error_for_n": {str(n): error_for(str(n)) for n in n_list},
                "pool_tag": tag,
                "error_kwargs": ERROR_KWARGS,
                "horizons": list(rcfg["horizons"]), "window_t": ts,
                "obs_weeks": [str(pd.Timestamp(d).date()) for d in weeks],
                "param_names": PARAM_NAMES}
        with open(os.path.join(out_dir, "rolling_meta.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="System B rolling selection")
    ap.add_argument("--tag", default="seir_1M_0308")
    ap.add_argument("--rebuild-pool", action="store_true")
    ap.add_argument("--pool-only", action="store_true", help="build the pool and stop")
    ap.add_argument("--n-history", nargs="+", default=None,
                    help="only these n_history values, e.g. --n-history all")
    ap.add_argument("--error-for-all", default=None,
                    help="distance to use for n_history=all, e.g. biased_mape_nz")
    ap.add_argument("--out-tag", default=None,
                    help="write the selection under this tag instead of --tag")
    a = ap.parse_args()

    build_pool(a.tag, a.rebuild_pool)
    if not a.pool_only:
        print()
        run_rolling(a.tag, only_n=a.n_history,
                    error_for_all=a.error_for_all, out_tag=a.out_tag)
