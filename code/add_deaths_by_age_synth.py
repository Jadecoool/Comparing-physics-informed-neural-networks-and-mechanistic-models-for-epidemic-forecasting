"""
Recover the deaths-by-age of a synthetic ABC-SMC run.

    SYNTH_SCEN=R0_2.0_noise_light python add_deaths_by_age_synth.py
    SYNTH_SCEN=R0_2.0_noise_light SMC_WEEK=12 python add_deaths_by_age_synth.py

abc_smc_synth.py did not store the per-age deaths, but it did store the seed of
every accepted particle, and SEIR() starts with np.random.seed(seed). So the
posterior plus the seed gives back the identical trajectory and the age split
can just be simulated again. No calibration is redone.

The check is the weekly new infections: the replay must match trajectories.npz
exactly, integer for integer. If it does, the deaths in the same run are the
ones the fit really implied.

Note the synthetic data are new infections, so deaths never entered the
distance. These are a model by-product of the fitted (R0, inf_t0), with
Delta = 0 and the London ifr table.

Writes deaths_by_age.npz into the window folder:
    weekly_age   (particles, 16, 35)
    seeds        (particles,)
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from compartment_model_age_deaths_SEIR import SEIR
from abc_seir_S import (C, DAILY_STEPS, DEATHS_DELAY, DETECTION_RATE, EPS, MU,
                        Nk, OUTPUT, REC_T0, SEASONALITY_MIN, HEMISPHERE, ifr)


SCENARIO = os.environ.get("SYNTH_SCEN", "R0_2.0_noise_light")
N_HISTORY = os.environ.get("SMC_NHIST", "all")

NW = 35
NDAYS = 7 * NW
DELTA_FIXED = 0
START = datetime(2020, 1, 1)

root = OUTPUT + "/smc_synth/" + SCENARIO + "/n" + N_HISTORY

if "SMC_WEEK" in os.environ:
    WINDOWS = [int(os.environ["SMC_WEEK"])]
else:
    WINDOWS = sorted(int(name[1:]) for name in os.listdir(root)
                     if name.startswith("w"))

T = NDAYS + 1
dates = [START + timedelta(days=d) for d in range(T)]


def to_weeks(daily):
    """Daily rows -> 35 weekly columns, on the grid the run used.

    The scenarios prepend a zero, so their week j is our week j-1. Same shift
    abc_smc_synth.run_model applies to the infections.
    """
    drop = daily[:, :NDAYS].reshape(daily.shape[0], NW, 7).sum(axis=2)
    out = np.zeros((daily.shape[0], NW))
    out[:, 1:] = drop[:, :NW - 1]
    return out


def one_window(t):
    folder = root + "/w" + str(t)
    post = pd.read_csv(folder + "/posterior.csv")
    seeds = np.load(folder + "/seeds.npz")["seeds"]
    stored = np.load(folder + "/trajectories.npz")["trajectories"]

    n = len(post)
    weekly_age = np.zeros((n, 16, NW))
    bad = 0

    for i in range(n):
        results = SEIR(inf_t0=float(post["inf_t0"].values[i]), rec_t0=REC_T0,
                       Nk=Nk, T=T, R0=float(post["R0"].values[i]), eps=EPS,
                       mu=MU, ifr=ifr, Delta=DELTA_FIXED, C=C,
                       detection_rate=DETECTION_RATE, dates=dates,
                       hemisphere=HEMISPHERE, seasonality_min=SEASONALITY_MIN,
                       deaths_delay=DEATHS_DELAY, daily_steps=DAILY_STEPS,
                       seed=int(seeds[i]))

        inf = np.array(results["infections"], dtype=float)[None, :]
        if not np.array_equal(to_weeks(inf)[0], stored[i]):
            bad = bad + 1
            continue

        age = np.array(results["deaths_age"], dtype=float)
        weekly_age[i] = to_weeks(age)

    if bad > 0:
        print("w%-3d ABORT: %d of %d particles did not reproduce. Nothing written."
              % (t, bad, n))
        return False

    np.savez_compressed(folder + "/deaths_by_age.npz",
                        weekly_age=weekly_age, seeds=seeds)

    total = weekly_age.sum(axis=(1, 2))
    print("w%-3d %4d particles reproduced exactly   total deaths median %.0f"
          "   -> %s" % (t, n, np.median(total), folder + "/deaths_by_age.npz"))
    return True


def main():
    print("scenario  : " + SCENARIO)
    print("n_history : " + N_HISTORY)
    print("windows   : " + str(WINDOWS))
    print()

    done = 0
    for t in WINDOWS:
        if one_window(t):
            done = done + 1

    print()
    print(str(done) + " of " + str(len(WINDOWS)) + " windows written")


if __name__ == "__main__":
    main()
