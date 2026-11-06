"""
Configuration + data loading for the ABC-with-budget pipeline.

Everything that the simulator needs but that does NOT vary between candidates
(contact matrices, IFR, population, the weekly binning, the observed data) is
resolved exactly once here and handed to the simulator as plain numpy arrays.

This is the single biggest speed win over the original code: the original
`SEIR()` rebuilt the daily contact matrices with ~170 pandas lookups on EVERY
call, which is most of its 0.71 s/sample runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "london_abc.yaml")

FREEZE_DATE = datetime(2020, 7, 5)  # legacy cut-off used by imply_reductions()


# -----------------------------------------------------------------------------
# config file
# -----------------------------------------------------------------------------
def load_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(path: str) -> str:
    """Make a config path absolute relative to the code/ directory."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


# -----------------------------------------------------------------------------
# the resolved setup
# -----------------------------------------------------------------------------
@dataclass
class Setup:
    """Everything the simulator and the selection code need, pre-computed."""

    cfg: Dict[str, Any]

    # demography / epidemiology
    Nk: np.ndarray                 # (A,)      population by age group
    ifr: np.ndarray                # (A,)      infection fatality rate by age group
    C: np.ndarray                  # (A, A)    baseline contact matrix
    Cs: np.ndarray                 # (T, A, A) daily contact matrix (reductions applied)
    eig0: float                    # leading eigenvalue of the day-0 next-gen matrix

    eps: float
    mu: float
    daily_steps: int

    # time axis
    start_org_date: datetime
    start_simu_date: datetime
    end_date: datetime
    dates: pd.DatetimeIndex        # (T,)  every simulated day
    week_ends: pd.DatetimeIndex    # (W_full,) week-ending Sunday of every bin
    week_starts: np.ndarray        # (W_full,) index of the first day of each bin
    cut_index: int                 # first bin that is compared against data

    # observation
    obs_total: np.ndarray          # (W,)      observed weekly total deaths
    obs_by_age: np.ndarray         # (W, A)    observed weekly deaths by age group
    obs_weeks: pd.DatetimeIndex    # (W,)      week-ending Sunday of each kept bin

    # convenience
    T: int = field(init=False)     # simulated days
    W: int = field(init=False)     # weekly points kept (== len(obs_total))
    A: int = field(init=False)     # number of age groups

    def __post_init__(self) -> None:
        self.T = len(self.dates)
        self.W = len(self.obs_total)
        self.A = len(self.Nk)

    # -- helpers --------------------------------------------------------------
    def week_index(self, date) -> int:
        """Index (into obs_total) of the week ending on `date`."""
        return int(self.obs_weeks.get_loc(pd.Timestamp(date)))

    def describe(self) -> str:
        return (
            f"basin={self.cfg['simulation']['basin']}  "
            f"days={self.T} ({self.dates[0].date()} .. {self.dates[-1].date()})  "
            f"weeks={self.W} ({self.obs_weeks[0].date()} .. {self.obs_weeks[-1].date()})  "
            f"age_groups={self.A}"
        )


# -----------------------------------------------------------------------------
# builder
# -----------------------------------------------------------------------------
def build_setup(cfg: Optional[Dict[str, Any]] = None,
                config_path: str = DEFAULT_CONFIG) -> Setup:
    cfg = cfg if cfg is not None else load_config(config_path)

    sim = cfg["simulation"]
    mdl = cfg["model"]
    paths = cfg["paths"]

    basin = sim["basin"]
    data_root = _resolve(paths["data"])

    start_org = datetime.strptime(sim["start_org_date"], "%Y-%m-%d")
    start_simu = datetime.strptime(sim["start_simu_date"], "%Y-%m-%d")
    end_date = datetime.strptime(sim["end_date"], "%Y-%m-%d")

    # ---- daily time axis ----------------------------------------------------
    # end_date is INCLUSIVE so the last weekly bin holds a full 7 days; see the
    # note in london_abc.yaml for why the legacy behaviour differed.
    if sim.get("legacy_end_date_exclusive", False):
        n_days = (end_date - start_org).days
    else:
        n_days = (end_date - start_org).days + 1
    dates = pd.date_range(start_org, periods=n_days, freq="D")

    # ---- weekly binning (identical to df.resample("W")) ---------------------
    # pandas "W" == "W-SUN": each bin is labelled by the Sunday that closes it.
    week_ends = pd.Series(0.0, index=dates).resample("W").sum().index
    day_bin = np.searchsorted(week_ends.values, dates.values, side="left")
    # first day index of every bin -> lets us aggregate with np.add.reduceat
    week_starts = np.searchsorted(day_bin, np.arange(len(week_ends)), side="left")

    if start_simu not in week_ends:
        raise ValueError(
            f"start_simu_date {start_simu.date()} is not a week-ending Sunday of the "
            f"simulation grid; the weekly bins close on {list(week_ends[:6].date)}..."
        )
    cut_index = int(week_ends.get_loc(pd.Timestamp(start_simu)))

    # ---- demography / epidemiology -----------------------------------------
    Nk = pd.read_csv(f"{data_root}/regions/{basin}/demographic/pop.csv")["population"].values
    Nk = np.asarray(Nk, dtype=np.float64)
    ifr = np.loadtxt(_resolve(paths["ifr"]), delimiter=",").astype(np.float64)
    C = np.loadtxt(_resolve(paths["contact_matrix"]), delimiter=",").astype(np.float64)

    if C.shape[0] != len(Nk) or len(ifr) != len(Nk):
        raise ValueError(
            f"shape mismatch: Nk={len(Nk)}, C={C.shape}, ifr={len(ifr)}"
        )

    # ---- daily contact matrices (reductions applied) ------------------------
    # `reductions_after_20200705` is the old name for this key; it only ever took
    # "data"/"freeze", so it cannot express mobility: none.
    mobility = mdl.get("mobility", mdl.get("reductions_after_20200705", "data"))
    omega = _build_omega(data_root, basin, dates, mode=mobility)
    Cs = np.ascontiguousarray(omega[:, None, None] * C[None, :, :])

    # beta is defined off the day-0 next-generation matrix, exactly as in
    # functions_general.get_beta(); the eigenvalue is constant across candidates
    # so it is computed once here and beta = R0 * mu / eig0.
    C_hat = (Nk[:, None] / Nk[None, :]) * Cs[0]
    eig0 = float(np.max(np.linalg.eigvals(C_hat).real))

    # ---- observed weekly deaths --------------------------------------------
    # paths.weekly_deaths lets a run point at a different target series (e.g. a
    # total-only file covering a longer window). Falls back to the age-resolved
    # file that ships with the basin.
    deaths_path = _resolve(paths.get(
        "weekly_deaths", f"{data_root}/regions/{basin}/epidemic/weekly_deaths.csv"))
    deaths_name = os.path.basename(deaths_path)
    deaths_df = pd.read_csv(deaths_path)
    deaths_df["date"] = pd.to_datetime(deaths_df["date"])
    deaths_df = deaths_df.set_index("date").sort_index()

    kept_weeks = week_ends[cut_index:]
    missing = kept_weeks.difference(deaths_df.index)
    if len(missing):
        raise ValueError(
            f"{deaths_name} is missing {len(missing)} of the {len(kept_weeks)} scoring "
            f"weeks, first {missing[0].date()} last {missing[-1].date()}. Either move "
            f"simulation.start_simu_date/end_date inside the file's range "
            f"({deaths_df.index[0].date()} .. {deaths_df.index[-1].date()}) or point "
            f"paths.weekly_deaths at a file that covers them."
        )
    obs = deaths_df.loc[kept_weeks]
    obs_total = obs["total"].values.astype(np.float64)

    # A total-only target file is fine; abc.target must then stay "total".
    age_cols = [c for c in obs.columns if c != "total"]
    obs_by_age = obs[age_cols].values.astype(np.float64)

    if len(age_cols) == 0:
        if cfg.get("abc", {}).get("target") == "by_age":
            raise ValueError(
                f"abc.target is 'by_age' but {deaths_name} has no age columns; "
                f"point paths.weekly_deaths at the age-resolved file."
            )
    elif obs_by_age.shape[1] != len(Nk):
        raise ValueError(
            f"{deaths_name} has {obs_by_age.shape[1]} age columns but pop.csv has {len(Nk)}"
        )

    return Setup(
        cfg=cfg, Nk=Nk, ifr=ifr, C=C, Cs=Cs, eig0=eig0,
        eps=float(mdl["eps"]), mu=float(mdl["mu"]),
        daily_steps=int(sim["daily_steps"]),
        start_org_date=start_org, start_simu_date=start_simu, end_date=end_date,
        dates=dates, week_ends=week_ends, week_starts=week_starts, cut_index=cut_index,
        obs_total=obs_total, obs_by_age=obs_by_age, obs_weeks=kept_weeks,
    )


def _build_omega(data_root: str, basin: str, dates: pd.DatetimeIndex,
                 mode: str = "data") -> np.ndarray:
    """
    Daily contact-reduction multiplier, vectorised.

    Reproduces functions_general.imply_reductions():
      mode="freeze" -> after 2020-07-05 the 2020-07-05 value is held constant
                       (the legacy NYC synthetic-second-wave setting)
      mode="data"   -> the real daily values are used throughout
                       (all_reductions.csv covers 2020-02-15 .. 2020-12-31)
      mode="none"   -> omega == 1 every day, i.e. a CONSTANT contact matrix.
                       all_reductions.csv is not read at all, so end_date is not
                       limited by its coverage.
    """
    if mode == "none":
        return np.ones(len(dates), dtype=np.float64)

    red = pd.read_csv(f"{data_root}/regions/{basin}/restriction/all_reductions.csv")
    red["date"] = pd.to_datetime(red["date"])
    series = red.set_index("date")["all_red"].sort_index()

    if mode == "freeze":
        frozen = float(series.loc[FREEZE_DATE])
        lookup = series.reindex(dates)
        lookup[dates > FREEZE_DATE] = frozen
    elif mode == "data":
        lookup = series.reindex(dates)
    else:
        raise ValueError(f"unknown mobility mode {mode!r} (use 'none', 'data' or 'freeze')")

    if lookup.isna().any():
        gaps = dates[lookup.isna().values]
        raise ValueError(
            f"all_reductions.csv has no value for {len(gaps)} simulated day(s), "
            f"first missing: {gaps[0].date()}, last: {gaps[-1].date()}. "
            f"Either shorten simulation.end_date or set "
            f"model.reductions_after_20200705: 'freeze'."
        )
    return lookup.values.astype(np.float64)


# -----------------------------------------------------------------------------
# run metadata
# -----------------------------------------------------------------------------
def dump_meta(path: str, setup: Setup, extra: Optional[Dict[str, Any]] = None) -> None:
    meta = {
        "config": setup.cfg,
        "created": datetime.now().isoformat(timespec="seconds"),
        "T_days": setup.T,
        "W_weeks": setup.W,
        "A_groups": setup.A,
        "dates": [str(setup.dates[0].date()), str(setup.dates[-1].date())],
        "obs_weeks": [str(d.date()) for d in setup.obs_weeks],
        "eig0": setup.eig0,
    }
    if extra:
        meta.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    s = build_setup()
    print(s.describe())
    print(f"eig0 = {s.eig0:.4f}   -> beta = R0 * mu / eig0")
    print(f"omega range: {s.Cs[:, 0, 0].min() / s.C[0, 0]:.3f} .. "
          f"{s.Cs[:, 0, 0].max() / s.C[0, 0]:.3f}")
    print(f"observed total deaths, first 5 weeks: {s.obs_total[:5]}")
    print(f"observed total deaths, last 5 weeks:  {s.obs_total[-5:]}")
