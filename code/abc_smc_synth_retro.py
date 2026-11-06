"""
Retrospective ABC-SMC fit of a synthetic scenario. One fit, no rolling windows.

    SYNTH_SCEN=R0_2.0_noise_light python abc_smc_synth_retro.py

Data are weekly new infections (S -> E). How many weeks are fitted depends on
the scenario: R0 = 1.5 uses the first 34, R0 = 2.0 the first 24.

For the final 1000 particles the weekly infections AND the weekly deaths are
saved both as a total and split by the 16 age groups.

The distance is biased_wmape with the collaborator's weights (offset=0): a
recency-weighted mean of |log(y+1) - log(yhat+1)| over every fitted week. No
week is dropped and nothing is floored -- the data are used as they come.

delta_t is fixed at 0 and Delta at 0, so only R0 and inf_t0 are fitted. Saves
the posterior and, for the final 1000 particles, the weekly infections AND the
weekly deaths over the whole 35-week grid.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyabc
from pyabc import RV, Distribution

from abc_errors import biased_wmape
from compartment_model_age_deaths_SEIR import SEIR
from abc_seir_S import (C, DAILY_STEPS, DEATHS_DELAY, DETECTION_RATE, EPS, MU,
                        NTOT, Nk, OUTPUT, REC_T0, SEASONALITY_MIN, HEMISPHERE,
                        ifr)


# ----------------------------- settings -------------------------------------
SCENARIO = os.environ.get("SYNTH_SCEN", "R0_2.0_noise_light")

NW = 35                  # weeks in every scenario
NDAYS = 7 * NW

# how many weeks go into the fit: 34 for R0 = 1.5, 24 for R0 = 2.0, whose tail
# is long and mostly empty. The saved trajectories always span all 35 weeks --
# only the distance is restricted to CAL -- so the weeks past NFIT are there to
# plot as hold-out.
if SCENARIO.startswith("R0_1.5"):
    NFIT = 34
else:
    NFIT = 24
CAL = slice(0, NFIT)

# Log, every week kept as it comes, and the collaborator's weights (newest week
# gets 1, offset=0). Nothing is dropped or floored: the synthetic data get no
# special treatment. Week 0 is 0 in every scenario and 0 in the model too, so it
# contributes exactly 0 to the sum while still counting in the denominator.
ERROR_NAME = "biased_wmape (offset=0)"
WEIGHT_OFFSET = 0.0

R0_MIN, R0_MAX = 1.0, 3.0             # truth is 1.5 or 2.0
I0_MIN, I0_MAX = 1.0, 10000.0         # log-uniform, truth is 1000
DELTA_FIXED = 0

POPULATION_SIZE = int(os.environ.get("SMC_POP", 1000))
MAX_GENERATIONS = int(os.environ.get("SMC_GENS", 10))
N_PROCS = int(os.environ.get("SMC_NPROCS", 1))
MAX_WALLTIME_HOURS = float(os.environ.get("SMC_HOURS", 11))
MINIMUM_EPSILON = 0.3

# SEIR wants a date list, but seasonality_min = 1.0 makes the dynamics ignore it
START = datetime(2020, 1, 1)
# -----------------------------------------------------------------------------


here = os.path.dirname(os.path.abspath(__file__))
data_file = here + "/../data/synthetic/" + SCENARIO + ".csv"
out_folder = OUTPUT + "/smc_synth_retro/" + SCENARIO

all_inc = pd.read_csv(data_file)["incidence"].values.astype(float)
observations = all_inc[CAL]

T = NDAYS + 1
dates = [START + timedelta(days=d) for d in range(T)]


def weekly_from_daily(daily):
    """Daily values -> 35 weekly sums, on the grid the scenarios use.

    The scenarios prepend a zero, so their week j is our week j-1.
    """
    drop = np.asarray(daily, dtype=float)[:NDAYS].reshape(NW, 7).sum(axis=1)
    out = np.zeros(NW)
    out[1:] = drop[:NW - 1]
    return out


def weekly_age_from_daily(daily):
    """Same, for a (16, days) array: -> (16, 35)."""
    a = np.asarray(daily, dtype=float)[:, :NDAYS]
    drop = a.reshape(a.shape[0], NW, 7).sum(axis=2)
    out = np.zeros((a.shape[0], NW))
    out[:, 1:] = drop[:, :NW - 1]
    return out


def run_model(R0, inf_t0, seed=None):
    """Weekly infections, deaths, deaths by age and susceptibles, 35 weeks."""
    if seed is None:
        seed = np.random.randint(0, 2 ** 31 - 1)
    seed = int(seed)

    results = SEIR(inf_t0=float(inf_t0), rec_t0=REC_T0, Nk=Nk, T=T,
                   R0=float(R0), eps=EPS, mu=MU, ifr=ifr, Delta=DELTA_FIXED,
                   C=C, detection_rate=DETECTION_RATE, dates=dates,
                   hemisphere=HEMISPHERE, seasonality_min=SEASONALITY_MIN,
                   deaths_delay=DEATHS_DELAY, daily_steps=DAILY_STEPS, seed=seed)

    infections = weekly_from_daily(results["infections"])
    infections_age = weekly_age_from_daily(results["infections_age"])
    deaths = weekly_from_daily(results["deaths"])
    deaths_age = weekly_age_from_daily(results["deaths_age"])
    # S at day 7j, i.e. the start of the week this grid calls week j
    S_weekly = np.asarray(results["S"], dtype=float)[0::7][:NW]
    return infections, infections_age, deaths, deaths_age, S_weekly, seed


def model(p):
    (infections, infections_age, deaths, deaths_age,
     S_weekly, seed) = run_model(p["R0"], p["inf_t0"])
    return {"data": infections[CAL], "all_weeks": infections,
            "infections_age": infections_age,
            "deaths": deaths, "deaths_age": deaths_age,
            "S_weekly": S_weekly, "seed": seed}


def distance(sim_data, actual_data):
    obs = np.array(actual_data["data"], dtype=float)
    sim = np.array(sim_data["data"], dtype=float)
    return float(biased_wmape(obs, sim[None, :], shift=1.0,
                             offset=WEIGHT_OFFSET)[0])


def main():
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print("scenario  : " + SCENARIO)
    print("fitting   : weeks 0.." + str(NFIT - 1) + "  (" + str(NFIT)
          + " of " + str(NW) + " data points)")
    print("distance  : " + ERROR_NAME)
    print("population: " + str(POPULATION_SIZE) + ", max "
          + str(MAX_GENERATIONS) + " generations, " + str(N_PROCS) + " cores")
    print()

    prior = Distribution(R0=RV("uniform", R0_MIN, R0_MAX - R0_MIN),
                         inf_t0=RV("loguniform", I0_MIN, I0_MAX))

    if N_PROCS > 1:
        from pyabc.sampler import MulticoreEvalParallelSampler
        sampler = MulticoreEvalParallelSampler(n_procs=N_PROCS)
    else:
        sampler = pyabc.sampler.SingleCoreSampler()

    abc = pyabc.ABCSMC(model, prior, distance,
                       transitions=pyabc.MultivariateNormalTransition(),
                       population_size=POPULATION_SIZE, sampler=sampler)

    db_path = os.path.join(out_folder, "history.db").replace("\\", "/")
    abc.new("sqlite:///" + db_path, {"data": observations})

    history = abc.run(minimum_epsilon=MINIMUM_EPSILON,
                      max_nr_populations=MAX_GENERATIONS,
                      max_walltime=timedelta(hours=MAX_WALLTIME_HOURS))

    particles = history.get_population().particles
    posterior = pd.DataFrame([dict(p.parameter) for p in particles])
    weights = np.array([p.weight for p in particles], dtype=float)
    distances = np.array([p.distance for p in particles], dtype=float)
    trajectories = np.array([p.sum_stat["all_weeks"] for p in particles])
    deaths = np.array([p.sum_stat["deaths"] for p in particles])
    deaths_age = np.array([p.sum_stat["deaths_age"] for p in particles])
    infections_age = np.array([p.sum_stat["infections_age"]
                               for p in particles])
    S_weekly = np.array([p.sum_stat["S_weekly"] for p in particles])
    seeds = np.array([p.sum_stat["seed"] for p in particles], dtype=np.int64)
    Rt = posterior["R0"].values[:, None] * S_weekly / NTOT

    posterior.to_csv(out_folder + "/posterior.csv", index=False)
    np.savez_compressed(out_folder + "/weights.npz",
                        weights=weights, distances=distances)
    np.savez_compressed(out_folder + "/trajectories.npz",
                        trajectories=trajectories)
    np.savez_compressed(out_folder + "/deaths.npz", deaths=deaths)
    np.savez_compressed(out_folder + "/deaths_by_age.npz",
                        weekly_age=deaths_age)
    np.savez_compressed(out_folder + "/infections_by_age.npz",
                        weekly_age=infections_age)
    np.savez_compressed(out_folder + "/S_weekly.npz",
                        S_weekly=S_weekly, N_total=NTOT)
    np.savez_compressed(out_folder + "/Rt.npz", Rt=Rt, N_total=NTOT)
    np.savez_compressed(out_folder + "/seeds.npz", seeds=seeds)
    history.get_all_populations().to_csv(out_folder + "/populations.csv",
                                         index=False)

    print()
    for name in ["R0", "inf_t0"]:
        v = posterior[name].values
        print("  %-8s median %10.3f   5-95%% [%.3f, %.3f]"
              % (name, np.median(v), np.percentile(v, 5),
                 np.percentile(v, 95)))
    print("  distance median %.4f" % np.median(distances))
    print("  weekly deaths total, median over particles %.0f"
          % np.median(deaths.sum(axis=1)))
    print("  deaths by age %s, age rows sum back to the total: %s"
          % (str(deaths_age.shape),
             np.allclose(deaths_age.sum(axis=1), deaths)))
    print("  infections by age %s, age rows sum back to the total: %s"
          % (str(infections_age.shape),
             np.allclose(infections_age.sum(axis=1), trajectories)))
    print()
    print("-> " + out_folder)


if __name__ == "__main__":
    main()
