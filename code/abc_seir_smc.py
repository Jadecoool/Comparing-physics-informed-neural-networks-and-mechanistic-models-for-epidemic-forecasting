"""
ABC-SMC for system B (SEIR null model, no mobility), on ONE rolling window.

Written to follow abcsmc_general.py: same pyabc calls, same rv_discrete priors,
same AggregatedTransition, same way of saving results.

One run does ONE window of ONE n_history setting. The window is week LAST_WEEK
of the 25-week grid (2020-02-16 .. 2020-08-02) and n_history says how far back
from it the scoring goes: "all" scores weeks 0..LAST_WEEK, "4" scores the last
four weeks, and so on. Which weeks exactly is not decided here -- it comes from
abc_select.calibration_slice, the same function abc_seir_rolling.py uses, so
SMC and ABC-with-budget always score the identical set of weeks and the two
posteriors can be put side by side.

    python abc_seir_smc.py

Change the settings below and run the file. There is no command line.
"""

import os
import numpy as np
import pandas as pd
import pyabc
from pyabc import RV, Distribution
from datetime import timedelta

from abc_errors import (biased_mape, biased_mape_nz, biased_wmape,
                        biased_wmape_nz, wmape)
from abc_select import calibration_slice
from compartment_model_age_deaths_SEIR import SEIR

# All the constants that decide what a trajectory looks like come from
# abc_seir_S, which is the file that already showed this exact set reproduces
# the pool. Do not re-type them here.
from abc_seir_S import (C, DAILY_STEPS, DEATHS_DELAY, DETECTION_RATE, END_DATE,
                        EPS, EVAL_START, MU, NTOT, NW, Nk, OUTPUT, REC_T0,
                        SEASONALITY_MIN, SEED_REF, HEMISPHERE, ifr)


# ----------------------------- settings -------------------------------------
POOL_TAG = "seir_1M_0216"     # which budget-ABC run this is compared against

# The folder the results are written to. It defaults to POOL_TAG, so a normal run
# is unchanged. Give it a name of its own when something in THIS file has changed
# -- a prior, say -- and the old results have to be kept:
#
#     SMC_RUNTAG=seir_1M_0216_i0max1000 python abc_seir_smc.py
#
# POOL_TAG still says which pool the figures compare against; RUN_TAG only says
# where to write. Without it a re-run silently overwrites the old one in place.
RUN_TAG = os.environ.get("SMC_RUNTAG", POOL_TAG)

# The window's "now" week. The rolling run does windows 3..22, so this goes
# 3..22 as well; week 22 is 2020-07-19. The job script sets it from
# SLURM_ARRAY_TASK_ID, one window per array task, so all 20 run at the same time
# as independent jobs.
LAST_WEEK = int(os.environ.get("SMC_WEEK", 22))

# How many weeks back from LAST_WEEK are scored: "4", "5", "6" or "all".
# The job script sets it, one submission per setting.
N_HISTORY = os.environ.get("SMC_NHIST", "all")

# Which distance goes with which n_history. This is a COPY of the table in
# abc_seir_rolling.py -- if that one changes, change this one too, or SMC and
# ABC-with-budget stop being comparable.
ERROR_FOR_N = {"4": "wmape", "5": "wmape", "6": "wmape", "all": "biased_wmape"}
if N_HISTORY not in ERROR_FOR_N:
    raise SystemExit("SMC_NHIST must be one of " + str(sorted(ERROR_FOR_N))
                     + ", got " + repr(N_HISTORY))
# Two more things that belong to the RUN rather than to the n_history, both
# keyed by RUN_TAG and both defaulting to what every earlier run did, so no older
# tag changes behaviour:
#
#   ERROR_FOR_ALL  overrides the distance used for n_history="all"
#   FIRST_WEEK     earliest week allowed into the scoring slice
#
# For seir_1M_0216_mape_from0315 the two MUST travel together. biased_mape has
# no log shift and divides by the observed count, so it cannot score weeks 0..3,
# where London recorded no deaths at all. Week 4 is 2020-03-15, the first week
# with any (37).
# seir_1M_0216_wmape_from0308 is plain wmape (no log, no recency weights) scored
# from week 3, 2020-03-08. wmape divides by the SUM of the observed weeks, so the
# zero in week 3 is harmless to it, unlike biased_mape.
# seir_1M_0216_mape_skipzero keeps the whole weeks 0..t slice and lets the
# distance drop the zero weeks itself, so FIRST_WEEK stays 0 for it.
ERROR_FOR_ALL = {"seir_1M_0216_mape_from0315": "biased_mape",
                 "seir_1M_0216_mape_skipzero": "biased_mape_nz",
                 "seir_logwmape_skipzero": "biased_wmape_nz",
                 "seir_1M_0216_wmape_from0308": "wmape"}
FIRST_WEEK = {"seir_1M_0216_mape_from0315": 4,
              "seir_1M_0216_wmape_from0308": 3}

# Tags whose ERROR_FOR_ALL entry applies to EVERY n_history, not just "all".
# Without this a submission with SMC_NHIST=5 would quietly fall back to the
# plain wmape of ERROR_FOR_N and land in this tag's folder, which would be the
# baseline run under another name. Leaving a tag out of this set keeps the old
# behaviour, so no finished run changes meaning.
ERROR_FOR_EVERY_N = {"seir_logwmape_skipzero"}


def error_for(run_tag, n_history):
    """Which distance a given run used for a given n_history."""
    if run_tag in ERROR_FOR_ALL:
        if n_history == "all" or run_tag in ERROR_FOR_EVERY_N:
            return ERROR_FOR_ALL[run_tag]
    return ERROR_FOR_N[n_history]


ERROR_NAME = error_for(RUN_TAG, N_HISTORY)
FIRST_SCORED_WEEK = FIRST_WEEK.get(RUN_TAG, 0)


def first_week_for(run_tag):
    """Earliest week a given run let into its scoring slice."""
    return FIRST_WEEK.get(run_tag, 0)


# The priors of each run, picked by RUN_TAG.
#
# THE POINT OF THE TABLE IS THAT THIS FILE NEVER HAS TO BE EDITED BETWEEN TWO
# SUBMISSIONS. An array task reads this file at the moment it STARTS, not when it
# was submitted, so editing the priors while an earlier array still has tasks in
# the queue would quietly hand those tasks the new priors and mix two runs into
# one folder, with nothing in the output to show it. With the table, the same file
# serves every run and both arrays can sit in the queue together.
#
# A tag that is not in the table raises KeyError rather than running with a guess.
#
# The first row is the pool's own prior (pools_B/seir_1M_0216/meta.json). Every
# other row changes it, and each one asks a different question. The first three
# NARROW a prior to see what the fit costs when an escape route is closed; the
# last three WIDEN or MOVE one to see whether the posterior really wanted to go
# somewhere the pool never let it.
#
# --- narrowing ---
#   R0max25            R0 [1.3, 2.5] instead of [1.3, 3.0]. Under the wide prior
#                      the n_history="all" posterior sat on the 3.0 ceiling in 9
#                      of the 20 windows (medians 2.8-2.9 at t=5..8, t=14..19).
#                      Cost turned out to be near zero: R0 dropped to ~1.7 and
#                      delta_t went to ~84 instead, an equally good fit.
#   i0max1000          inf_t0 [1, 1000] instead of [1, 10000]. That posterior sat
#                      against the 10000 ceiling from t=5 to t=13 -- medians
#                      6800-8200 and even the 5th percentile at 4000-5600, so ALL
#                      of it was above 1000 in those windows.
#   R0max25_i0max1000  both at once. R0 and inf_t0 trade off against each other --
#                      a lower R0 can be paid for by seeding more, and the other
#                      way round -- so narrowing one alone leaves that escape route
#                      open, and only this row closes it. The cost came out
#                      SUPER-ADDITIVE in 14 of the 17 informative windows.
#
# --- widening / moving ---
#   i0max50000         inf_t0 [1, 50000] instead of [1, 10000]. The mirror image
#                      of i0max1000: the baseline posterior was pressed against
#                      the 10000 ceiling, so this asks how much further up it
#                      would have gone if allowed. 50000 is 0.56% of NTOT.
#   R0max25_i0max50000 R0 held down while inf_t0 is let up. If the two really do
#                      trade off, this is the combination that should fit BEST of
#                      all -- more seeds paying for the lower R0.
#   dtmin30            delta_t [30, 90] instead of [0, 90], i.e. seeding no later
#                      than 2020-02-07, with R0 and inf_t0 left at the pool's own
#                      values. This is a MOVE, not a widening: it forbids the late
#                      seeding the early windows chose (delta_t medians 18-27 at
#                      t=5..13) and forces the early-seeding branch everywhere, so
#                      the two branches can be compared at equal footing.
#
# Every row after the first differs from the pool, so they all need
# SMC_ALLOW_PRIOR_DRIFT=1, and the budget-ABC lines in their figures come from a
# pool drawn with the pool's own priors -- not the same prior as the run. That
# belongs in the caption.
#
# The Delta and delta_t bounds are the numpy kind, upper end EXCLUSIVE: (7, 36) is
# the integers 7..35 and (0, 91) is 0..90.
PRIORS_FOR_RUN = {
    #                                   R0           inf_t0 (log)    Delta    delta_t
    "seir_1M_0216":                    ((1.3, 3.0), (1.0, 10000.0), (7, 36), (0, 91)),
    "seir_1M_0216_R0max25":            ((1.3, 2.5), (1.0, 10000.0), (7, 36), (0, 91)),
    "seir_1M_0216_i0max1000":          ((1.3, 3.0), (1.0,  1000.0), (7, 36), (0, 91)),
    "seir_1M_0216_R0max25_i0max1000":  ((1.3, 2.5), (1.0,  1000.0), (7, 36), (0, 91)),
    "seir_1M_0216_i0max50000":         ((1.3, 3.0), (1.0, 50000.0), (7, 36), (0, 91)),
    "seir_1M_0216_R0max25_i0max50000": ((1.3, 2.5), (1.0, 50000.0), (7, 36), (0, 91)),
    "seir_1M_0216_dtmin30":            ((1.3, 3.0), (1.0, 10000.0), (7, 36), (30, 91)),
    "seir_1M_0216_mape_from0315":      ((1.3, 3.0), (1.0, 10000.0), (7, 36), (0, 91)),
    "seir_1M_0216_wmape_from0308":     ((1.3, 3.0), (1.0, 10000.0), (7, 36), (0, 91)),
    "seir_1M_0216_mape_skipzero":      ((1.3, 3.0), (1.0, 10000.0), (7, 36), (0, 91)),
    "seir_logwmape_skipzero":  ((1.3, 3.0), (1.0, 10000.0), (7, 36), (0, 91)),
}

prior_set = PRIORS_FOR_RUN[RUN_TAG]
R0_MIN, R0_MAX = prior_set[0]
I0_MIN, I0_MAX = prior_set[1]                  # log-uniform
DELTA_MIN, DELTA_MAX = prior_set[2]            # integers 7..35
DELTA_T_MIN, DELTA_T_MAX = prior_set[3]        # integers 0..90

# These four can be set from the environment so the job script does not have to
# edit this file. Same idea as ABC_N / ABC_NJOBS in abc_seir.py.
POPULATION_SIZE = int(os.environ.get("SMC_POP", 1000))
MAX_GENERATIONS = int(os.environ.get("SMC_GENS", 10))
N_PROCS = int(os.environ.get("SMC_NPROCS", 1))
# Keep this BELOW the SLURM --time. pyabc then stops at a generation boundary and
# returns normally, so the save block below runs. If SLURM kills the job instead,
# history.db still has the finished generations but posterior.csv and
# trajectories.npz are never written.
MAX_WALLTIME_HOURS = float(os.environ.get("SMC_HOURS", 6))

MINIMUM_EPSILON = 0.3         # stopping condition. budget ABC got about 0.81
P_STAY = 0.7                  # for the discrete jump transitions
# N_PROCS = 1 means SingleCoreSampler. More than 1 uses
# MulticoreEvalParallelSampler, which forks -- Linux only, so on Windows leave
# it at 1.
# -----------------------------------------------------------------------------


here = os.path.dirname(os.path.abspath(__file__))
data_file = here + "/../data/regions/London/epidemic/weekly_deaths_total.csv"
# One folder per n_history AND per window. Both levels are needed: without "/w22"
# every array task would overwrite the others, and without "/nall" the n=4 run of
# window 22 would overwrite the n=all run of window 22. RUN_TAG is the level above
# both, so a run with different settings does not overwrite an older one either.
out_folder = (OUTPUT + "/smc_B/" + RUN_TAG
              + "/n" + N_HISTORY + "/w" + str(LAST_WEEK))


# ---- the observed deaths ----
table = pd.read_csv(data_file)
all_deaths = table["total"].values.astype(float)
all_dates = pd.to_datetime(table["date"]).values

# The weeks this window scores. Taken from abc_select rather than worked out
# here, so it is the same slice abc_seir_rolling.py stored as cal_lo.
CAL = calibration_slice(LAST_WEEK, N_HISTORY, FIRST_SCORED_WEEK)
observations = all_deaths[CAL]

# A window scoring fewer than two weeks constrains nothing: with a single week
# every candidate that lands on that one number is accepted and the posterior
# comes back as the prior. With FIRST_SCORED_WEEK = 4 that rules out t = 3 and
# t = 4, which is why this run is submitted as --array=5-22.
# How many weeks this window really scores. biased_mape_nz throws the zero weeks
# away, so for it the slice length overstates the count.
if ERROR_NAME.endswith("_nz"):
    N_SCORED = int((observations > 0).sum())
else:
    N_SCORED = len(observations)

# One scored week pins nothing down: every candidate that lands on that single
# number is accepted and the posterior comes back as the prior. Two is the
# minimum everywhere EXCEPT biased_mape_nz, where the single-point window is
# asked for on purpose -- its first window is the first week with any deaths.
MIN_SCORED = 1 if ERROR_NAME.endswith("_nz") else 2
if N_SCORED < MIN_SCORED:
    raise SystemExit("window %d scores %d week(s), fewer than the %d this"
                     " distance needs." % (LAST_WEEK, N_SCORED, MIN_SCORED))
if N_SCORED == 1:
    print("WARNING: this window scores ONE week. Any candidate matching that one"
          " number is accepted, so read its posterior as barely-updated prior.")

# biased_mape divides by the observed count, so a zero week is not merely
# uninformative, it is undefined. Say so here rather than quietly flooring it.
if ERROR_NAME == "biased_mape" and (observations <= 0).any():
    raise SystemExit("biased_mape cannot score a week with zero deaths, and"
                     " weeks %d..%d contain one" % (CAL.start, CAL.stop - 1))

# Only now, once this window is known to be usable. Making it earlier leaves an
# empty w<t>/ behind for every window the guards above reject, and the plotting
# scripts read the folder listing to find out which windows a run has.
if not os.path.exists(out_folder):
    os.makedirs(out_folder)


def run_model(R0, inf_t0, Delta, delta_t, seed=None, want_age=False):
    """
    Run one candidate and put its weekly deaths, and the susceptibles left at
    the end of each week, on the fixed 25-week grid.

    seed=None draws a fresh one, which is what the fitting wants: two particles
    with the same parameters should still get different noise. Passing a seed back
    in reproduces that particle EXACTLY -- SEIR() starts with np.random.seed(seed)
    -- which is how add_deaths_by_age_smc.py recovers the per-age deaths later,
    without this run having to store a (16, 25) array for every particle.

    Returns the weekly deaths, the weekly susceptibles, and the seed used. With
    want_age=True it returns a fourth thing, the same weekly deaths split by age
    group, shape (16, 25). The fitting never asks for it -- storing a (16, 25)
    array per particle would bloat history.db -- but it is put here rather than in
    add_deaths_by_age_smc.py so that the zero-padding rule has only ONE copy and
    the age rows can never end up on a different grid than the total.

    A candidate starts on SEED_REF - delta_t, so each one simulates a different
    number of days. The grid is always the same 25 weeks, so the result is
    padded with zeros in front (if it started after the grid opens) or cut at
    the front (if it started before). This is the same thing abc_seir.py and
    abc_seir_S.py do.
    """
    start_org = SEED_REF - timedelta(days=int(delta_t))
    T = (END_DATE - start_org).days
    dates = [start_org + timedelta(days=d) for d in range(T)]

    # A new random seed every time, unless the caller supplied one.
    if seed is None:
        seed = np.random.randint(0, 2 ** 31 - 1)
    seed = int(seed)

    results = SEIR(inf_t0=float(inf_t0), rec_t0=REC_T0, Nk=Nk, T=T, R0=float(R0),
                   eps=EPS, mu=MU, ifr=ifr, Delta=int(Delta), C=C,
                   detection_rate=DETECTION_RATE, dates=dates,
                   hemisphere=HEMISPHERE, seasonality_min=SEASONALITY_MIN,
                   deaths_delay=DEATHS_DELAY, daily_steps=DAILY_STEPS, seed=seed)

    daily = np.array(results["deaths"], dtype=float)
    daily_S = np.array(results["S"], dtype=float)   # susceptibles, one per day

    offset = (start_org - EVAL_START).days
    if offset >= 0:
        on_grid = np.concatenate([np.zeros(offset), daily])
        # Before this candidate is seeded nobody has been infected yet, so the
        # padding for S is the WHOLE population, not zero. Getting this wrong
        # makes Rt nonsense for every candidate with delta_t < 26.
        on_grid_S = np.concatenate([np.full(offset, NTOT), daily_S])
    else:
        on_grid = daily[-offset:]
        on_grid_S = daily_S[-offset:]

    weekly = on_grid.reshape(NW, 7).sum(axis=1)
    # Day 7j+6 is the Sunday that closes week j, which is the day the rolling
    # windows are labelled with. Same rule as abc_seir_S.py.
    S_weekly = on_grid_S[6::7]
    if not want_age:
        return weekly, S_weekly, seed

    # The same padding again, one row per age group. deaths_age is (16, T) and is
    # what results["deaths"] was summed from, so the rows must add back up to
    # weekly -- add_deaths_by_age_smc.py checks exactly that.
    age = np.array(results["deaths_age"], dtype=float)
    if offset >= 0:
        on_grid_age = np.concatenate(
            [np.zeros((len(age), offset)), age], axis=1)
    else:
        on_grid_age = age[:, -offset:]
    weekly_age = on_grid_age.reshape(len(age), NW, 7).sum(axis=2)
    return weekly, S_weekly, seed, weekly_age


def model(p):
    """What pyabc calls. 'data' is what the distance uses."""
    weekly, S_weekly, seed = run_model(p["R0"], p["inf_t0"], p["Delta"],
                                       p["delta_t"])
    # Only "data" is scored. The rest ride along so nothing has to be guessed at
    # afterwards. "seed" is the important one: with it a particle's run can be
    # repeated exactly, so anything the model computed but did not store here --
    # the per-age deaths, for instance -- can still be recovered later. Without it
    # the draw is gone and that particle can never be reproduced.
    return {"data": weekly[CAL],
            "all_weeks": weekly,
            "S_weekly": S_weekly,
            "seed": seed}


def distance(sim_data, actual_data):
    """
    The error function for this n_history, wrapped for pyabc.

    Both come from abc_errors, the same module abc_select uses, so an SMC
    distance and a budget-ABC error on the same weeks are the same number.
    Note the two are NOT on the same scale as each other -- wmape is a ratio of
    sums, biased_wmape a weighted mean |log ratio| -- so never compare an n=4
    distance with an n=all one.
    """
    obs = np.array(actual_data["data"], dtype=float)
    sim = np.array(sim_data["data"], dtype=float)
    if ERROR_NAME == "biased_wmape":
        return float(biased_wmape(obs, sim[None, :], shift=1.0)[0])
    if ERROR_NAME == "biased_mape":
        return float(biased_mape(obs, sim[None, :])[0])
    if ERROR_NAME == "biased_mape_nz":
        return float(biased_mape_nz(obs, sim[None, :])[0])
    if ERROR_NAME == "biased_wmape_nz":
        return float(biased_wmape_nz(obs, sim[None, :], shift=1.0)[0])
    return float(wmape(obs, sim[None, :])[0])


def make_prior():
    """The priors of pools_B/seir_1M_0216, written the pyabc way."""
    n_delta = DELTA_MAX - DELTA_MIN
    n_delta_t = DELTA_T_MAX - DELTA_T_MIN

    prior = Distribution(
        R0=RV("uniform", R0_MIN, R0_MAX - R0_MIN),
        inf_t0=RV("loguniform", I0_MIN, I0_MAX),
        Delta=RV("rv_discrete", values=(
            np.arange(DELTA_MIN, DELTA_MAX),
            [1. / n_delta] * n_delta)),
        delta_t=RV("rv_discrete", values=(
            np.arange(DELTA_T_MIN, DELTA_T_MAX),
            [1. / n_delta_t] * n_delta_t)),
    )
    return prior


def make_transition():
    """How a generation is perturbed to propose the next one."""
    transition = pyabc.AggregatedTransition(mapping={
        "R0": pyabc.MultivariateNormalTransition(),
        "inf_t0": pyabc.MultivariateNormalTransition(),
        "Delta": pyabc.DiscreteJumpTransition(
            domain=np.arange(DELTA_MIN, DELTA_MAX), p_stay=P_STAY),
        "delta_t": pyabc.DiscreteJumpTransition(
            domain=np.arange(DELTA_T_MIN, DELTA_T_MAX), p_stay=P_STAY),
    })
    return transition


def make_sampler():
    if N_PROCS > 1:
        from pyabc.sampler import MulticoreEvalParallelSampler
        return MulticoreEvalParallelSampler(n_procs=N_PROCS)
    return pyabc.sampler.SingleCoreSampler()


def main():
    print("ABC-SMC, system B (SEIR null, no mobility)")
    print("  run tag      : " + RUN_TAG
          + ("" if RUN_TAG == POOL_TAG
             else "   (figures compare against pool '" + POOL_TAG + "')"))
    print("  n_history    : " + N_HISTORY)
    print("  window       : now = week " + str(LAST_WEEK)
          + ", scores weeks " + str(CAL.start) + ".." + str(CAL.stop - 1)
          + "  (" + str(pd.Timestamp(all_dates[CAL.start]).date())
          + " .. " + str(pd.Timestamp(all_dates[LAST_WEEK]).date()) + ")")
    print("  observed     : " + str(len(observations)) + " weeks, "
          + str(int(observations.sum())) + " deaths")
    print("  distance     : " + ERROR_NAME
          + ("" if FIRST_SCORED_WEEK == 0
             else "   (scoring starts at week " + str(FIRST_SCORED_WEEK)
                  + ", " + str(pd.Timestamp(all_dates[FIRST_SCORED_WEEK]).date())
                  + ")"))
    print("  priors       : R0 [" + str(R0_MIN) + ", " + str(R0_MAX) + "]"
          + "  inf_t0 log [" + str(I0_MIN) + ", " + str(I0_MAX) + "]"
          + "  Delta [" + str(DELTA_MIN) + ", " + str(DELTA_MAX - 1) + "]"
          + "  delta_t [" + str(DELTA_T_MIN) + ", " + str(DELTA_T_MAX - 1) + "]")
    print("  population   : " + str(POPULATION_SIZE)
          + "   max generations " + str(MAX_GENERATIONS)
          + "   min epsilon " + str(MINIMUM_EPSILON))
    print("  cores        : " + str(N_PROCS))
    print()

    prior = make_prior()
    transition = make_transition()
    sampler = make_sampler()

    abc = pyabc.ABCSMC(model, prior, distance,
                       transitions=transition,
                       population_size=POPULATION_SIZE,
                       sampler=sampler)

    db_path = os.path.join(out_folder, "history.db").replace("\\", "/")
    abc.new("sqlite:///" + db_path, {"data": observations})

    history = abc.run(minimum_epsilon=MINIMUM_EPSILON,
                      max_nr_populations=MAX_GENERATIONS,
                      max_walltime=timedelta(hours=MAX_WALLTIME_HOURS))

    # ---- save ----
    # Everything comes out of ONE call. get_population gives each particle's
    # parameters, weight, summary statistics and distance together, so the rows
    # of posterior.csv, trajectories.npz, S_weekly.npz and Rt.npz line up by
    # construction. get_distribution() and get_weighted_sum_stats() are two
    # separate queries that sort differently, and Rt = R0 * S / N would be
    # silently wrong if their order ever disagreed.
    particles = history.get_population().particles

    posterior = pd.DataFrame([dict(p.parameter) for p in particles])
    weights = np.array([p.weight for p in particles], dtype=float)
    distances = np.array([p.distance for p in particles], dtype=float)
    trajectories = np.array([p.sum_stat["all_weeks"] for p in particles])
    S_weekly = np.array([p.sum_stat["S_weekly"] for p in particles])
    # One seed per particle, in the same row order as everything else. This is
    # what makes the run reproducible particle by particle.
    seeds = np.array([p.sum_stat["seed"] for p in particles], dtype=np.int64)

    # Rt = R0 * S(t) / N, the same definition add_Rt.py uses for the pool
    Rt = posterior["R0"].values[:, None] * S_weekly / NTOT

    posterior.to_csv(out_folder + "/posterior.csv", index=False)
    np.savez_compressed(out_folder + "/weights.npz",
                        weights=weights, distances=distances)
    np.savez_compressed(out_folder + "/trajectories.npz",
                        trajectories=trajectories)
    np.savez_compressed(out_folder + "/S_weekly.npz",
                        S_weekly=S_weekly, N_total=NTOT)
    np.savez_compressed(out_folder + "/Rt.npz", Rt=Rt, N_total=NTOT)
    np.savez_compressed(out_folder + "/seeds.npz", seeds=seeds)

    # how many simulations each generation cost
    populations = history.get_all_populations()
    populations.to_csv(out_folder + "/populations.csv", index=False)

    print()
    print(populations.to_string())
    print()
    print("total simulations: " + str(int(populations["samples"].sum())))
    print()
    print("posterior (last generation, " + str(len(posterior)) + " particles):")
    for name in ["R0", "inf_t0", "Delta", "delta_t"]:
        v = posterior[name].values
        print("  %-8s median %10.2f   5-95%% [%.2f, %.2f]"
              % (name, np.median(v), np.percentile(v, 5), np.percentile(v, 95)))

    print("  %-8s median %10.2f   5-95%% [%.2f, %.2f]"
          % ("Rt(wk" + str(LAST_WEEK) + ")", np.median(Rt[:, LAST_WEEK]),
             np.percentile(Rt[:, LAST_WEEK], 5),
             np.percentile(Rt[:, LAST_WEEK], 95)))
    print("  distance      best %.4f  median %.4f  worst %.4f"
          % (distances.min(), np.median(distances), distances.max()))
    ess = weights.sum() ** 2 / np.sum(weights ** 2)
    print("  effective sample size %.1f out of %d" % (ess, len(weights)))

    print()
    print("-> " + out_folder + "/posterior.csv")
    print("-> " + out_folder + "/weights.npz")
    print("-> " + out_folder + "/trajectories.npz")
    print("-> " + out_folder + "/S_weekly.npz")
    print("-> " + out_folder + "/Rt.npz")
    print("-> " + out_folder + "/seeds.npz")
    print("-> " + out_folder + "/populations.csv")
    print("-> " + out_folder + "/history.db")


if __name__ == "__main__":
    main()
