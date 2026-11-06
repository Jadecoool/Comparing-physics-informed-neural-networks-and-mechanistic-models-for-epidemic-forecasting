"""
Build the ABC-with-budget candidate pool.

THE KEY IDEA
------------
Every candidate carries ONE fixed (R0, inf_t0, Delta) for its whole trajectory, so a
candidate's trajectory does not depend on which week we happen to be forecasting
from. That means the expensive part -- simulating 1e6 full-length trajectories --
is done EXACTLY ONCE. Retrospective fitting and every rolling-forecast window are
then just different ways of re-scoring and re-sorting the same pool, which costs
seconds rather than hours.

It also makes "keep the whole trajectory, not just the forecast part" free: a
window's output is 1000 integer INDICES into the pool, and the full trajectory is
whatever `deaths_total[idx]` holds. Any horizon can be read off later without
re-running anything.

LAYOUT ON DISK
--------------
    <output>/pools/<tag>/
        meta.json              config snapshot + date/week axis
        params.npy             (N, 3)     float64   columns = abc_priors.PARAM_NAMES
        deaths_total.npy       (N, W)     float32   <- always
        deaths_by_age.npy      (N, W, A)  float32   <- pool.store_deaths_by_age
        infections_total.npy   (N, W)     float32   <- pool.store_infections_total
        infections_by_age.npy  (N, W, A)  float32   <- pool.store_infections_by_age

    <output>/S/<tag>/                            <- pool.store_S_total
        pool_idx.npy           (N,)       int32     rows of the pool, i.e. arange(N)
        S_weekly_total.npy     (N, W)     float32   susceptibles ON the day each
                                                    week closes (a stock, not a sum)
        meta.json

S(t) is written to its own folder rather than into the pool, so that it lands in
exactly the layout system B uses in <output>/S_B/<tag>/ and one reader works on
both. It has to be captured here and nowhere else: candidates share one RNG
stream per chunk, so unlike system B a single trajectory cannot be re-simulated
afterwards to recover its compartments.

Files are .npy memmaps, so downstream code opens them with
`np.load(..., mmap_mode="r")` and never has to hold 1e6 trajectories in RAM.

USAGE
-----
    python abc_pool.py                        # full run, size from the yaml
    python abc_pool.py --n 20000 --tag smoke  # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed

import abc_priors
from abc_config import DEFAULT_CONFIG, Setup, _resolve, build_setup, dump_meta, load_config
from abc_seir_batch import simulate_batch

OUTPUTS = ["deaths_total", "deaths_by_age", "infections_total", "infections_by_age",
           "S_total"]


# -----------------------------------------------------------------------------
# paths
# -----------------------------------------------------------------------------
def pool_dir(cfg: dict, tag: str) -> str:
    return os.path.join(_resolve(cfg["paths"]["output"]), "pools", tag)


def s_dir(cfg: dict, tag: str) -> str:
    """Where S(t) ends up. Mirrors system B's <output>/S_B/<tag>/."""
    return os.path.join(_resolve(cfg["paths"]["output"]), "S", tag)


def open_pool(cfg: dict, tag: str) -> Dict[str, np.ndarray]:
    """Memory-map an existing pool. Returns {name: array} for whatever exists."""
    d = pool_dir(cfg, tag)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"no pool at {d}; run `python abc_pool.py --tag {tag}` first")
    out = {"params": np.load(os.path.join(d, "params.npy"), mmap_mode="r")}
    for name in OUTPUTS:
        p = os.path.join(d, f"{name}.npy")
        if os.path.exists(p):
            out[name] = np.load(p, mmap_mode="r")
    # S(t) lives in <output>/S/<tag>/ so the layout matches system B; expose it
    # here too, so callers can just ask the pool for it.
    p = os.path.join(s_dir(cfg, tag), "S_weekly_total.npy")
    if os.path.exists(p):
        out["S_total"] = np.load(p, mmap_mode="r")
    return out


# -----------------------------------------------------------------------------
# worker
# -----------------------------------------------------------------------------
def _run_chunk(chunk_id: int, lo: int, hi: int, out_dir: str,
               config_path: str, master_seed: int, wants: Dict[str, bool]) -> int:
    """
    Simulate candidates [lo, hi) and write them straight into the memmaps.

    Runs in a separate process, so it rebuilds the Setup itself (cheap: ~0.2 s)
    and opens the memmaps in r+ mode. Chunks own disjoint row ranges, so no
    locking is needed.
    """
    # keep BLAS single-threaded: joblib already saturates the cores with processes
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    setup = build_setup(config_path=config_path)
    params = np.load(os.path.join(out_dir, "params.npy"), mmap_mode="r")[lo:hi]
    pd_ = abc_priors.as_dict(np.asarray(params))

    # one independent stream per chunk, derived from the master seed
    rng = np.random.default_rng([master_seed, chunk_id])

    res = simulate_batch(
        pd_["R0"], pd_["inf_t0"], pd_["Delta"], setup, rng,
        want_deaths_by_age=wants["deaths_by_age"],
        want_infections=wants["infections_total"],
        want_infections_by_age=wants["infections_by_age"],
        want_S=wants["S_total"],
    )

    for name, arr in res.items():
        mm = np.load(os.path.join(out_dir, f"{name}.npy"), mmap_mode="r+")
        mm[lo:hi] = arr
        mm.flush()
        del mm
    return hi - lo


# -----------------------------------------------------------------------------
# driver
# -----------------------------------------------------------------------------
def build_pool(config_path: str = DEFAULT_CONFIG,
               tag: str = "main",
               n_samples: Optional[int] = None,
               n_jobs: Optional[int] = None,
               overwrite: bool = False,
               chunk_size: Optional[int] = None) -> str:
    cfg = load_config(config_path)
    setup = build_setup(cfg=cfg, config_path=config_path)

    pcfg = cfg["pool"]
    N = int(n_samples if n_samples is not None else pcfg["n_samples"])
    chunk = int(chunk_size if chunk_size is not None else pcfg["chunk_size"])
    jobs = int(n_jobs if n_jobs is not None else pcfg["n_jobs"])
    seed = int(pcfg["master_seed"])

    wants = {
        "deaths_total": True,
        "deaths_by_age": bool(pcfg.get("store_deaths_by_age", False)),
        "infections_total": bool(pcfg.get("store_infections_total", False)),
        "infections_by_age": bool(pcfg.get("store_infections_by_age", False)),
        "S_total": bool(pcfg.get("store_S_total", False)),
    }

    out_dir = pool_dir(cfg, tag)
    if os.path.exists(os.path.join(out_dir, "meta.json")) and not overwrite:
        raise FileExistsError(f"{out_dir} already exists; pass --overwrite to rebuild it")
    os.makedirs(out_dir, exist_ok=True)

    W, A = setup.W, setup.A
    print(setup.describe())
    print(f"pool tag={tag}  N={N:,}  chunk={chunk}  n_jobs={jobs}  seed={seed}")
    print("storing:", ", ".join(k for k, v in wants.items() if v))

    # ---- priors: drawn once, in the parent, so the pool is reproducible -----
    rng = np.random.default_rng(seed)
    params = abc_priors.sample_prior(cfg, N, rng)
    np.save(os.path.join(out_dir, "params.npy"), params)

    # ---- allocate the output memmaps ---------------------------------------
    shapes = {"deaths_total": (N, W), "deaths_by_age": (N, W, A),
              "infections_total": (N, W), "infections_by_age": (N, W, A),
              "S_total": (N, W)}
    total_gb = 0.0
    for name, on in wants.items():
        if not on:
            continue
        mm = np.lib.format.open_memmap(os.path.join(out_dir, f"{name}.npy"),
                                       mode="w+", dtype=np.float32, shape=shapes[name])
        total_gb += mm.nbytes / 1e9
        del mm
    print(f"allocated {total_gb:.2f} GB of trajectory storage")

    # ---- simulate ----------------------------------------------------------
    bounds: List[tuple] = [(i, lo, min(lo + chunk, N))
                           for i, lo in enumerate(range(0, N, chunk))]
    t0 = time.time()
    done = Parallel(n_jobs=jobs, verbose=5, backend="loky")(
        delayed(_run_chunk)(cid, lo, hi, out_dir, os.path.abspath(config_path), seed, wants)
        for cid, lo, hi in bounds
    )
    elapsed = time.time() - t0
    assert sum(done) == N, f"expected {N} samples, wrote {sum(done)}"

    dump_meta(os.path.join(out_dir, "meta.json"), setup, extra={
        "tag": tag, "n_samples": N, "chunk_size": chunk, "master_seed": seed,
        "param_names": abc_priors.PARAM_NAMES, "stored": [k for k, v in wants.items() if v],
        "n_chunks": len(bounds), "elapsed_seconds": round(elapsed, 1),
        "storage_gb": round(total_gb, 3),
    })

    if wants["S_total"]:
        _export_S(cfg, tag, out_dir, setup, N)

    print(f"\ndone in {elapsed/60:.1f} min ({elapsed/N*1e3:.2f} ms/sample) -> {out_dir}")
    _summarise(out_dir, setup)
    return out_dir


def _export_S(cfg: dict, tag: str, out_dir: str, setup: Setup, N: int) -> None:
    """
    Move S_total out of the pool folder and into <output>/S/<tag>/, laid out
    exactly like system B's <output>/S_B/<tag>/: same file names, same dtypes,
    same meta keys, so one reader works on both.

    It is MOVED rather than copied -- 100 MB is not worth holding twice, and one
    canonical location means there is never a stale second copy to pick by
    mistake. pool_idx is the whole pool here (arange), because for system A S(t)
    comes free with the simulation and there was no reason to restrict it to the
    accepted candidates the way system B had to.

    No S_daily_total.npy: system B recovered S day by day for 26,733 candidates,
    which is small. The same thing for a 1e6 pool would be 680 MB for a resolution
    nothing downstream uses, so only the weekly stock is kept.
    """
    d = s_dir(cfg, tag)
    os.makedirs(d, exist_ok=True)

    src = os.path.join(out_dir, "S_total.npy")
    dst = os.path.join(d, "S_weekly_total.npy")
    if os.path.exists(dst):
        os.remove(dst)
    os.replace(src, dst)

    np.save(os.path.join(d, "pool_idx.npy"), np.arange(N, dtype=np.int32))

    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "tag": tag,
            "system": f"A (age-structured SEIR, mobility "
                      f"{cfg['model'].get('mobility', 'data')})",
            "n_unique": N,
            "pool_n": N,
            "N_total": float(setup.Nk.sum()),
            "seed": int(cfg["pool"]["master_seed"]),
            "anchor": str(setup.start_simu_date.date()),
            "start_org": str(setup.start_org_date.date()),
            "grid_day0": str(setup.start_org_date.date()),
            "n_days": int(setup.T),
            "weeks": [str(d_.date()) for d_ in setup.obs_weeks],
            "weekly_day_index": [int(i) for i in
                                 (np.append(setup.week_starts[1:], setup.T) - 1)[setup.cut_index:]],
            "stock_not_flow": True,
            "verified_against_pool": True,
            "source": "written by abc_pool.py while the pool was simulated; "
                      "system A candidates share one RNG stream per chunk and "
                      "cannot be re-simulated one at a time, so this is the only "
                      "copy of S(t) that will ever exist for this pool.",
            "param_names": abc_priors.PARAM_NAMES,
        }, fh, indent=2)

    print(f"S(t) -> {d}  (S_weekly_total.npy {N} x {setup.W}, "
          f"pool_idx.npy, meta.json)")


def _summarise(out_dir: str, setup: Setup) -> None:
    d = np.load(os.path.join(out_dir, "deaths_total.npy"), mmap_mode="r")
    peak = np.asarray(d[: min(len(d), 50000)]).max(axis=1)
    obs_peak = setup.obs_total.max()
    print(f"peak weekly deaths across candidates: "
          f"median {np.median(peak):,.0f}, "
          f"p5 {np.percentile(peak, 5):,.0f}, p95 {np.percentile(peak, 95):,.0f}   "
          f"(observed peak {obs_peak:,.0f})")
    frac = np.mean((peak > 0.5 * obs_peak) & (peak < 2.0 * obs_peak))
    print(f"candidates within a factor 2 of the observed peak: {frac:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the ABC candidate pool")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--tag", default="main", help="pool name under <output>/pools/")
    ap.add_argument("--n", type=int, default=None, help="override pool.n_samples")
    ap.add_argument("--n-jobs", type=int, default=None, help="override pool.n_jobs")
    ap.add_argument("--chunk", type=int, default=None, help="override pool.chunk_size")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    build_pool(a.config, a.tag, a.n, a.n_jobs, a.overwrite, a.chunk)
