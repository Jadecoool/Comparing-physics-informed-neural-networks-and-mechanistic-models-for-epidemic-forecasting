"""
ABC-SMC on the synthetic scenarios, one rolling window per run.

Data are weekly new infections (S -> E), 35 weeks. delta_t is fixed at 0, so
only R0 and inf_t0 are fitted.

    SYNTH_SCEN=R0_2.0_noise_light SMC_WEEK=12 SMC_NHIST=all python abc_smc_synth.py
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyabc
from pyabc import RV, Distribution

from abc_errors import biased_wmape, wmape
from abc_select import calibration_slice
from compartment_model_age_deaths_SEIR import SEIR
from abc_seir_S import (C, DAILY_STEPS, DEATHS_DELAY, DETECTION_RATE, EPS, MU,
                        NTOT, Nk, OUTPUT, REC_T0, SEASONALITY_MIN, HEMISPHERE,
                        ifr)


# ----------------------------- settings -------------------------------------
SCENARIO = os.environ.get("SYNTH_SCEN", "R0_2.0_noise_light")
LAST_WEEK = int(os.environ.get("SMC_WEEK", 12))
N_HISTORY = os.environ.get("SMC_NHIST", "all")

NW = 35                  # weeks in every scenario
NDAYS = 7 * NW

ERROR_FOR_N = {"4": "wmape", "5": "wmape", "6": "wmape", "all": "biased_wmape"}
ERROR_NAME = ERROR_FOR_N[N_HISTORY]

R0_MIN, R0_MAX = 1.0, 3.0            # truth is 1.5 or 2.0
I0_MIN, I0_MAX = 1.0, 10000.0        # log-uniform, truth is 1000
DELTA_FIXED = 0                      # only feeds deaths, which we do not use

POPULATION_SIZE = int(os.environ.get("SMC_POP", 1000))
MAX_GENERATIONS = int(os.environ.get("SMC_GENS", 10))
N_PROCS = int(os.environ.get("SMC_NPROCS", 1))
MAX_WALLTIME_HOURS = float(os.environ.get("SMC_HOURS", 6))
MINIMUM_EPSILON = 0.3

# SEIR wants a date list, but seasonality_min = 1.0 makes the dynamics ignore it
START = datetime(2020, 1, 1)
# -----------------------------------------------------------------------------


here = os.path.dirname(os.path.abspath(__file__))
data_file = here + "/../data/synthetic/" + SCENARIO + ".csv"
out_folder = (OUTPUT + "/smc_synth/" + SCENARIO
              + "/n" + N_HISTORY + "/w" + str(LAST_WEEK))

all_inc = pd.read_csv(data_file)["incidence"].values.astype(float)
CAL = calibration_slice(LAST_WEEK, N_HISTORY)
observations = all_inc[CAL]


def run_model(R0, inf_t0, seed=None):
    """Weekly new infections and weekly susceptibles, 35 weeks each."""
    T = NDAYS + 1
    dates = [START + timedelta(days=d) for d in range(T)]

    if seed is None:
        seed = np.random.randint(0, 2 ** 31 - 1)
    seed = int(seed)

    results = SEIR(inf_t0=float(inf_t0), rec_t0=REC_T0, Nk=Nk, T=T,
                   R0=float(R0), eps=EPS, mu=MU, ifr=ifr, Delta=DELTA_FIXED,
                   C=C, detection_rate=DETECTION_RATE, dates=dates,
                   hemisphere=HEMISPHERE, seasonality_min=SEASONALITY_MIN,
                   deaths_delay=DEATHS_DELAY, daily_steps=DAILY_STEPS, seed=seed)

    daily_inf = np.array(results["infections"], dtype=float)[:NDAYS]
    drop = daily_inf.reshape(NW, 7).sum(axis=1)

    # the scenarios prepend a zero, so their week j is our week j-1
    weekly = np.concatenate([[0.0], drop[:NW - 1]])
    S_weekly = np.array(results["S"], dtype=float)[7::7]
    return weekly, S_weekly, seed


def model(p):
    weekly, S_weekly, seed = run_model(p["R0"], p["inf_t0"])
    return {"data": weekly[CAL], "all_weeks": weekly,
            "S_weekly": S_weekly, "seed": seed}


def distance(sim_data, actual_data):
    obs = np.array(actual_data["data"], dtype=float)
    sim = np.array(sim_data["data"], dtype=float)
    if ERROR_NAME == "biased_wmape":
        return float(biased_wmape(obs, sim[None, :], shift=1.0)[0])
    return float(wmape(obs, sim[None, :])[0])


def main():
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print("scenario  : " + SCENARIO)
    print("window    : now = week " + str(LAST_WEEK) + ", scores weeks "
          + str(CAL.start) + ".." + str(CAL.stop - 1))
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
    S_weekly = np.array([p.sum_stat["S_weekly"] for p in particles])
    seeds = np.array([p.sum_stat["seed"] for p in particles], dtype=np.int64)
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
    history.get_all_populations().to_csv(out_folder + "/populations.csv",
                                         index=False)

    print()
    for name in ["R0", "inf_t0"]:
        v = posterior[name].values
        print("  %-8s median %10.3f   5-95%% [%.3f, %.3f]"
              % (name, np.median(v), np.percentile(v, 5),
                 np.percentile(v, 95)))
    print("  distance median %.4f" % np.median(distances))
    print()
    print("-> " + out_folder)


if __name__ == "__main__":
    main()
