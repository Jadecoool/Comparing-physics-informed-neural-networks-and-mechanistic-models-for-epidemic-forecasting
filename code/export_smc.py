"""
Export the results of an ABC-SMC run out of its history.db.

    python export_smc.py

abc_seir_smc.py already writes posterior.csv, weights.npz, trajectories.npz and
populations.csv when it finishes normally, so in the normal case you never need
this file. It is for the two cases where that does not happen:

  1. The job was killed -- SLURM hit its time limit, the node died, a generation
     never filled up. pyabc writes history.db after EVERY generation, and
     max_nr_recorded_particles is inf by default, so the database already holds
     every finished generation with all its particles AND their trajectories.
     Only the generation that was in progress is lost. Run this afterwards and
     nothing else is.

  2. The job is still running and you want to see how far it has got. sqlite
     allows readers while pyabc writes, so this is safe; if it happens to land
     exactly on a write you get "database is locked" and can just run it again.

It writes the same four files with the same names as abc_seir_smc.py, so
plot_smc_B.py works either way and does not need to know which one produced them.
"""

import os
import numpy as np
import pandas as pd
import pyabc

# Only for N, to turn S(t) into Rt. It is the same number abc_seir_smc.py used.
from abc_seir_S import NTOT


# ----------------------------- settings -------------------------------------
POOL_TAG = "seir_1M_0216"     # must match abc_seir_smc.py

# Which run to export. The same two environment variables abc_seir_smc.py reads,
# so inside a job script it picks the right one on its own. From the command
# line, say which one you want:
#
#     SMC_WEEK=15 SMC_NHIST=4 python export_smc.py
#
LAST_WEEK = int(os.environ.get("SMC_WEEK", 22))
N_HISTORY = os.environ.get("SMC_NHIST", "all")

# Which folder to look in. The same variable abc_seir_smc.py uses, with the same
# default, so a normal run needs nothing here. If the run was given a RUN_TAG of
# its own, give this the same one or it will not find the database.
RUN_TAG = os.environ.get("SMC_RUNTAG", POOL_TAG)
# -----------------------------------------------------------------------------


here = os.path.dirname(os.path.abspath(__file__))
folder = (here + "/../output/smc_B/" + RUN_TAG
          + "/n" + N_HISTORY + "/w" + str(LAST_WEEK))
db_file = os.path.abspath(folder + "/history.db")

print("run      : tag " + RUN_TAG + ", n_history " + N_HISTORY
      + ", window now = week " + str(LAST_WEEK))
if not os.path.exists(db_file):
    print("Cannot find " + db_file)
    print("Nothing to export -- run abc_seir_smc.py first.")
    raise SystemExit(1)

# sqlite wants forward slashes even on Windows
db_url = "sqlite:///" + db_file.replace("\\", "/")
history = pyabc.History(db_url)

# A database can hold several runs if abc.new() was called more than once on it.
# Take the newest.
runs = history.all_runs()
history.id = runs[-1].id
print("database : " + db_file)
print("runs     : " + str(len(runs)) + ", using id " + str(history.id))

populations = history.get_all_populations()
# t = -1 is the calibration step, not a real generation
finished = populations[populations["t"] >= 0]

if len(finished) == 0:
    print("No generation has finished yet -- nothing to export.")
    raise SystemExit(0)

last_t = int(finished["t"].max())
print("finished : " + str(len(finished)) + " generations, last is t = "
      + str(last_t))
print()
print(populations.to_string())
print()
print("total simulations so far: " + str(int(populations["samples"].sum())))
print()

# ---- the same files abc_seir_smc.py writes ----
# One call, so parameters / weights / summary statistics belong to the same
# particle row by row. See the comment in abc_seir_smc.py for why this is not
# done with get_distribution() plus get_weighted_sum_stats().
particles = history.get_population(t=last_t).particles

posterior = pd.DataFrame([dict(p.parameter) for p in particles])
weights = np.array([p.weight for p in particles], dtype=float)
distances = np.array([p.distance for p in particles], dtype=float)
trajectories = np.array([p.sum_stat["all_weeks"] for p in particles])

posterior.to_csv(folder + "/posterior.csv", index=False)
np.savez_compressed(folder + "/weights.npz",
                    weights=weights, distances=distances)
np.savez_compressed(folder + "/trajectories.npz", trajectories=trajectories)

# One seed per particle, so the run can be repeated particle by particle. Runs
# made before the seed was stored do not have it and cannot get it afterwards.
if "seed" in particles[0].sum_stat:
    seeds = np.array([p.sum_stat["seed"] for p in particles], dtype=np.int64)
    np.savez_compressed(folder + "/seeds.npz", seeds=seeds)
    print("seeds   : %d stored, all different: %s"
          % (len(seeds), len(set(seeds.tolist())) == len(seeds)))
else:
    print("seeds   : this database has no seed, so particles cannot be repeated.")
populations.to_csv(folder + "/populations.csv", index=False)

# S(t), and Rt from it. Runs made before S_weekly was added do not have it, and
# it cannot be recovered afterwards -- run_model draws a fresh random seed every
# call and does not keep it, so a particle's trajectory cannot be reproduced.
if "S_weekly" in particles[0].sum_stat:
    S_weekly = np.array([p.sum_stat["S_weekly"] for p in particles])
    Rt = posterior["R0"].values[:, None] * S_weekly / NTOT
    np.savez_compressed(folder + "/S_weekly.npz",
                        S_weekly=S_weekly, N_total=NTOT)
    np.savez_compressed(folder + "/Rt.npz", Rt=Rt, N_total=NTOT)
    print("Rt      : week " + str(LAST_WEEK) + " median %.3f  5-95%% [%.3f, %.3f]"
          % (np.median(Rt[:, LAST_WEEK]),
             np.percentile(Rt[:, LAST_WEEK], 5),
             np.percentile(Rt[:, LAST_WEEK], 95)))
else:
    print("Rt      : this database has no S_weekly, so Rt cannot be computed.")
    print("          It was run before S(t) was stored. Re-run that window.")

print("exported generation t = " + str(last_t) + ": "
      + str(len(posterior)) + " particles, trajectories "
      + str(trajectories.shape))
print("distance: best %.4f  median %.4f  worst %.4f"
      % (distances.min(), np.median(distances), distances.max()))

# ESS says how many of those particles are really contributing. If it is far
# below the population size the posterior is narrower in appearance than in
# substance, and that belongs in any figure caption made from these files.
w = weights
ess = float(w.sum() ** 2 / np.sum(w ** 2))
print("effective sample size: %.1f out of %d" % (ess, len(posterior)))

print()
# NOTE these ignore the weights, so they are only a rough look. Figures should
# resample by weight first, the way plot_smc_B.py does.
print("unweighted quantiles:")
print("%-10s %10s %10s %10s" % ("param", "median", "5%", "95%"))
for name in ["R0", "inf_t0", "Delta", "delta_t"]:
    if name not in posterior.columns:
        continue
    v = posterior[name].values
    print("%-10s %10.2f %10.2f %10.2f"
          % (name, np.median(v), np.percentile(v, 5), np.percentile(v, 95)))

print()
print("-> " + folder + "/posterior.csv")
print("-> " + folder + "/weights.npz")
print("-> " + folder + "/trajectories.npz")
print("-> " + folder + "/populations.csv")
if os.path.exists(folder + "/Rt.npz"):
    print("-> " + folder + "/S_weekly.npz")
    print("-> " + folder + "/Rt.npz")
