"""
Recover the infections-by-age of the retrospective synthetic fits.

    python add_infections_by_age_retro.py                        # all four
    SYNTH_SCEN=R0_2.0_noise_light python add_infections_by_age_retro.py

abc_smc_synth_retro.py stored the age split of the deaths but not of the
infections. It did store the seed of every accepted particle, and SEIR() starts
with np.random.seed(seed), so the posterior plus the seed gives back the
identical trajectory and the age split can simply be simulated again. No
calibration is redone.

The check is the total: the 16 age rows must sum back to trajectories.npz
exactly, integer for integer. If any particle disagrees nothing is written.

Writes infections_by_age.npz into the scenario folder:
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


SCENARIOS = ["R0_1.5_noise_light", "R0_1.5_noise_mild",
             "R0_2.0_noise_light", "R0_2.0_noise_mild"]
if "SYNTH_SCEN" in os.environ:
    SCENARIOS = [os.environ["SYNTH_SCEN"]]

NW = 35
NDAYS = 7 * NW
DELTA_FIXED = 0
START = datetime(2020, 1, 1)

root = OUTPUT + "/smc_synth_retro"

T = NDAYS + 1
dates = [START + timedelta(days=d) for d in range(T)]


def weekly_age_from_daily(daily):
    """(16, days) -> (16, 35), on the grid the scenarios use.

    The scenarios prepend a zero, so their week j is our week j-1.
    """
    a = np.asarray(daily, dtype=float)[:, :NDAYS]
    drop = a.reshape(a.shape[0], NW, 7).sum(axis=2)
    out = np.zeros((a.shape[0], NW))
    out[:, 1:] = drop[:, :NW - 1]
    return out


def one_scenario(scen):
    folder = root + "/" + scen
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

        age = weekly_age_from_daily(results["infections_age"])
        if not np.array_equal(age.sum(axis=0), stored[i]):
            bad = bad + 1
            continue
        weekly_age[i] = age

    if bad > 0:
        print("%-20s ABORT: %d of %d particles did not reproduce. "
              "Nothing written." % (scen, bad, n))
        return False

    np.savez_compressed(folder + "/infections_by_age.npz",
                        weekly_age=weekly_age, seeds=seeds)
    print("%-20s %4d particles reproduced exactly   -> %s"
          % (scen, n, folder + "/infections_by_age.npz"))
    return True


def main():
    done = 0
    for scen in SCENARIOS:
        if one_scenario(scen):
            done = done + 1

    print()
    print(str(done) + " of " + str(len(SCENARIOS)) + " scenarios written")


if __name__ == "__main__":
    main()
