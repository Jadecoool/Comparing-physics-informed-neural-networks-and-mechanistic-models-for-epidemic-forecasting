#!/bin/bash
#SBATCH --job-name=seir_smc
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=3-22

# --array=3-22 : ONE TASK PER ROLLING WINDOW. 3..22 is exactly the window_t of
# abc_seir_rolling.py, so the 20 tasks line up one-for-one with the budget-ABC
# rolling result. They are independent jobs and run at the same time.
#
# Each task asks for 16 cores for 12 h, so the array as a whole reserves
# 20 x 16 x 12 = 3840 CPU-hours, while the real cost is nearer 150. SLURM
# prioritises on what you ASK for, so this can sit in the queue a long time.
# Adding %N runs at most N tasks at once, which schedules much more easily:
#
#     sbatch --array=3-22%5 job_abc_seir_smc.sh
#
# --partition=compute : not the computeshort every other job script here uses,
# because computeshort caps at 1 h and cannot hold a 12 h run. This pairing is
# proven -- window 22 ran to completion under it. For a short trial instead:
#
#     sbatch --partition=computeshort --time=01:00:00 --array=22 job_abc_seir_smc.sh

# =============================================================================
# SYSTEM B (SEIR null, no mobility): ABC-SMC on EVERY rolling window.
#
#   sbatch job_abc_seir_smc.sh                     # all 20 windows, n_history=all
#   sbatch --array=22 job_abc_seir_smc.sh          # just the last window
#
# ONE SUBMISSION PER n_history. The array covers the windows; n_history is a
# separate axis, set from the environment, one submission each:
#
#   sbatch --export=ALL,SMC_NHIST=all job_abc_seir_smc.sh
#   sbatch --export=ALL,SMC_NHIST=4   job_abc_seir_smc.sh
#   sbatch --export=ALL,SMC_NHIST=5   job_abc_seir_smc.sh
#   sbatch --export=ALL,SMC_NHIST=6   job_abc_seir_smc.sh
#
# A RUN WITH DIFFERENT SETTINGS GOES IN ITS OWN FOLDER. SMC_RUNTAG names it, and
# defaults to the pool tag, so the plain commands above are unchanged. When the
# priors in abc_seir_smc.py no longer match the pool the pre-flight refuses to
# start; SMC_ALLOW_PRIOR_DRIFT=1 says the difference is on purpose:
#
#   sbatch --export=ALL,SMC_NHIST=all,SMC_RUNTAG=seir_1M_0216_i0max1000,\
#                   SMC_ALLOW_PRIOR_DRIFT=1 job_abc_seir_smc.sh
#
# A RUN THAT SCORES FROM A LATER WEEK NEEDS A SHORTER ARRAY. Some runs do not
# score from the anchor: seir_1M_0216_mape_from0315 starts at week 4 (2020-03-15)
# because its distance, biased_mape, has no log shift and cannot take the zero
# deaths of weeks 0..3. Window 4 would then score a single week, which constrains
# nothing, so abc_seir_smc.py refuses it and the array must start at 5:
#
#   sbatch --array=5-22 --export=ALL,SMC_NHIST=all,\
#                  SMC_RUNTAG=seir_1M_0216_mape_from0315 job_abc_seir_smc.sh
#
# That one uses the pool own priors, so it needs NO SMC_ALLOW_PRIOR_DRIFT.
#
# (one line, no space after the commas). Results land in
# ../output/smc_B/seir_1M_0216_i0max1000/nall/w<t>/ and nothing older is touched.
#
# Four submissions of 20 tasks, 80 in total. Doing it this way rather than
# --array=0-79 with the n decoded from the task id keeps the arithmetic out of
# the script, and lets one n be re-run on its own if it goes wrong. %A differs
# per submission so the .out files never collide.
#
# Runs abc_seir_smc.py once per (n_history, window). Window t scores the weeks
# abc_select.calibration_slice(t, n) gives -- weeks 0..t for "all", the last n
# weeks otherwise -- with the error function abc_seir_rolling.py pairs with that
# n: wmape for 4/5/6, biased_wmape for "all". Same weeks, same metric as the
# ABC-with-budget rolling run on the 1e6 pool, so the posteriors compare window
# by window.
#
# Each task writes to its own folder, ../output/smc_B/<tag>/n<n>/w<t>/. Nothing
# is shared between tasks.
#
# Forecasts need no extra work: the model stores all 25 weeks of every accepted
# particle, so horizons 0..4 are just weeks t+1..t+5 of the saved trajectories.
#
# THIS IS NOT part of the pool pipeline. It does not read parts, does not build
# or touch a pool, and writes nothing any other script reads. It simulates from
# scratch, adaptively, and is safe to run alongside anything else.
#
# THE TWO TIME LIMITS MUST NOT BE EQUAL
#     --time=12:00:00   SLURM kills the job here
#     SMC_HOURS=11      pyabc stops itself an hour earlier and RETURNS
#
# That hour of headroom is the whole point. pyabc writes history.db after every
# generation, but abc_seir_smc.py only writes posterior.csv, trajectories.npz
# and populations.csv once abc.run() returns. Set both limits to 12 and pyabc
# never gets to return -- SLURM kills it mid-generation and those three files are
# never written. (Nothing is actually lost even then, see RECOVERY below, but it
# turns a finished run into a manual step.)
#
# pyabc checks the walltime BETWEEN generations. A generation whose acceptance
# rate has collapsed can run far past SMC_HOURS because the check never comes
# round, and SLURM kills it. That is the case the headroom cannot cover.
#
# RECOVERY -- nothing is ever wasted
# history.db holds every FINISHED generation, with all particles and their
# trajectories (max_nr_recorded_particles defaults to inf). Only the generation
# that was in progress is lost. After any kill:
#
#     python export_smc.py
#
# writes the same four files from the database. The same command also works
# WHILE the job is running, which is the way to see whether epsilon is still
# falling -- if two generations in a row barely move it, scancel is better than
# burning the remaining hours.
#
# COST, AND WHY TO WATCH IT
# Window 22 has been run: 10 generations, 135,020 simulations, 33 min on 16
# cores, about 7.5 CPU-hours. Simulation cost does not depend on the window (the
# trajectory length is set by delta_t, not by t), so the spread between windows
# comes only from acceptance rates. Twenty windows should land near 150
# CPU-hours in total -- against 31-55 CPU-hours for the whole 1e6 pool.
#
# The 12 h limit is insurance, roughly 20x the measured need. If a task really
# uses it, that task is worth looking at rather than trusting.
#
# Window 22 stopped on the generation limit at epsilon 1.0257, not on
# MINIMUM_EPSILON = 0.3, and its last four generations moved epsilon only 4-5%
# each. Expect the same for the later windows. Keeping SMC_GENS fixed at 10 is
# deliberate: every window then gets the same computational treatment, which is
# what makes the posterior widths comparable across windows.
# =============================================================================

set -uo pipefail

cd "$SLURM_SUBMIT_DIR"

module load python/3.8.5-gcc-12.2.0
source /data/home/ahw917/project_1_for_apocrita/proj1/bin/activate

# Which window this task calibrates. Inside an array job it comes from the task
# id; outside one it falls back to 22, and can be set by hand:
#     SMC_WEEK=15 sbatch --array=15 job_abc_seir_smc.sh
export SMC_WEEK="${SMC_WEEK:-${SLURM_ARRAY_TASK_ID:-22}}"

# How far back this run scores: "4", "5", "6" or "all". Set at submit time,
# one submission per setting (see the header). Not an array axis.
export SMC_NHIST="${SMC_NHIST:-all}"

# Which folder the results go in. Defaults to the pool tag, which is what
# abc_seir_smc.py defaults to as well, so leaving it alone changes nothing.
export SMC_RUNTAG="${SMC_RUNTAG:-seir_1M_0216}"

# Set to 1 ONLY when the priors in abc_seir_smc.py have been changed on purpose
# and no longer match the pool. See the pre-flight below.
export SMC_ALLOW_PRIOR_DRIFT="${SMC_ALLOW_PRIOR_DRIFT:-0}"

export SMC_POP="${SMC_POP:-1000}"
export SMC_GENS="${SMC_GENS:-10}"
export SMC_HOURS="${SMC_HOURS:-11}"
export SMC_NPROCS="${SLURM_CPUS_PER_TASK:-16}"

# Single-threaded numpy in every worker. pyabc forks SMC_NPROCS processes; if
# BLAS also opens threads inside each one they fight over the same cores. This
# also matters for correctness -- SEIR() seeds the global numpy RandomState, and
# BLAS threads inside a worker must not interleave with it.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

echo "host      : $(hostname)"
echo "partition : ${SLURM_JOB_PARTITION:-?}     time limit: ${SBATCH_TIMELIMIT:-see --time}"
echo "cores     : $SMC_NPROCS"
echo "run tag   : $SMC_RUNTAG   (prior drift allowed: $SMC_ALLOW_PRIOR_DRIFT)"
echo "n_history : $SMC_NHIST"
echo "window    : now = week $SMC_WEEK   (array task ${SLURM_ARRAY_TASK_ID:-none})"
echo "settings  : population $SMC_POP, max $SMC_GENS generations, pyabc stops at ${SMC_HOURS} h"
echo "python    : $(which python)  $(python -V 2>&1)"

# --- pre-flight ---------------------------------------------------------------
# Fail in seconds rather than after hours of sampling.
set -e
python - <<'PYCHECK'
import os, sys
import numpy as np
import pyabc
print("pyabc %s" % pyabc.__version__)

# MulticoreEvalParallelSampler forks, which Windows cannot do. Nothing here
# should ever run off a POSIX host, but say so rather than fail obscurely.
if os.name != "posix":
    sys.exit("ABORT: this job expects a POSIX host for the multicore sampler")

import abc_seir_smc as s
print("window    : now = week %d, scores weeks %d..%d (%d weeks, %d deaths)"
      % (s.LAST_WEEK, s.CAL.start, s.CAL.stop - 1,
         len(s.observations), int(s.observations.sum())))
print("n_history : %s   distance %s   scoring starts at week %d"
      % (s.N_HISTORY, s.ERROR_NAME, s.FIRST_SCORED_WEEK))
print("writing   : output/smc_B/%s/n%s/w%d" % (s.RUN_TAG, s.N_HISTORY, s.LAST_WEEK))
# The n and the distance must be paired the way abc_seir_rolling.py pairs them,
# or this run is not comparable with the budget-ABC selection it sits next to.
import abc_seir_rolling
if abc_seir_rolling.ERROR_FOR_N != s.ERROR_FOR_N:
    sys.exit("ABORT: ERROR_FOR_N differs from abc_seir_rolling.py\n  here: %s\n  there: %s"
             % (s.ERROR_FOR_N, abc_seir_rolling.ERROR_FOR_N))
print("errors    : ERROR_FOR_N matches abc_seir_rolling.py OK")
print("priors    : R0 [%g, %g]  inf_t0 log [%g, %g]  Delta [%d, %d]  delta_t [%d, %d]"
      % (s.R0_MIN, s.R0_MAX, s.I0_MIN, s.I0_MAX,
         s.DELTA_MIN, s.DELTA_MAX - 1, s.DELTA_T_MIN, s.DELTA_T_MAX - 1))
print("stopping  : min epsilon %s, max %d generations, %s h walltime"
      % (s.MINIMUM_EPSILON, s.MAX_GENERATIONS, s.MAX_WALLTIME_HOURS))
print("sampler   : %s on %d procs" % (type(s.make_sampler()).__name__, s.N_PROCS))

# The priors here are a copy of the pool's. If they have drifted, the two
# posteriors are not comparable and the point of the run is lost.
import json
meta = os.path.join(s.OUTPUT, "pools_B", s.POOL_TAG, "meta.json")
if os.path.exists(meta):
    want = {p[0]: (p[2], p[3]) for p in json.load(open(meta))["priors"]}
    have = {"R0": (s.R0_MIN, s.R0_MAX), "inf_t0": (s.I0_MIN, s.I0_MAX),
            "Delta": (s.DELTA_MIN, s.DELTA_MAX - 1),
            "delta_t": (s.DELTA_T_MIN, s.DELTA_T_MAX - 1)}
    bad = [k for k in want if tuple(map(float, want[k])) != tuple(map(float, have[k]))]
    if bad and os.environ.get("SMC_ALLOW_PRIOR_DRIFT", "0") != "1":
        sys.exit("ABORT: priors differ from pool '%s' on %s\n  pool: %s\n  here: %s\n"
                 "  If this is on purpose, re-submit with SMC_ALLOW_PRIOR_DRIFT=1 and a\n"
                 "  SMC_RUNTAG of its own, so the older results are not overwritten."
                 % (s.POOL_TAG, bad, want, have))
    # Different priors AND the old folder name would overwrite results that can no
    # longer be reproduced -- the run draws a fresh seed per particle and keeps none.
    if bad and s.RUN_TAG == s.POOL_TAG:
        sys.exit("ABORT: the priors differ from the pool but SMC_RUNTAG is still '%s'.\n"
                 "  This run would overwrite output/smc_B/%s/n%s/ in place.\n"
                 "  Give SMC_RUNTAG a new name." % (s.RUN_TAG, s.RUN_TAG, s.N_HISTORY))
    if bad:
        print("priors    : *** DIFFER FROM POOL '%s' ON %s -- ALLOWED ON PURPOSE ***"
              % (s.POOL_TAG, bad))
        for k in bad:
            print("            %-8s pool %s   this run %s" % (k, want[k], have[k]))
        print("            The budget-ABC lines in the figures come from that pool, so")
        print("            they are NOT this run's prior. Say so in the caption.")
    else:
        print("priors    : match pool '%s' OK" % s.POOL_TAG)
else:
    print("priors    : pool '%s' not on this machine, cannot cross-check. The SMC "
          "run still works; only the side-by-side comparison needs that pool."
          % s.POOL_TAG)

# One model call, so a broken import or a shape mistake shows up now.
w, sw, sd = s.run_model(R0=2.0, inf_t0=100.0, Delta=15, delta_t=45)
assert w.shape == (s.NW,), "deaths came back %s, expected (%d,)" % (w.shape, s.NW)
assert sw.shape == (s.NW,), "S came back %s, expected (%d,)" % (sw.shape, s.NW)
print("model     : one test call OK, %d weekly values, %.0f deaths" % (len(w), w.sum()))
# S must start at the full population and only ever fall. If the zero-padding
# rule were wrong it would start at 0 instead, and Rt would be nonsense.
assert sw[0] <= s.NTOT and (np.diff(sw) <= 0).all(), "S(t) is not a falling curve"
print("S(t)      : S/N goes %.4f -> %.4f, Rt(final) = %.3f"
      % (sw[0] / s.NTOT, sw[-1] / s.NTOT, 2.0 * sw[-1] / s.NTOT))
# Feeding the seed back must give the identical trajectory. If it does not, the
# per-age deaths cannot be recovered afterwards and seeds.npz is worthless, so
# find that out here rather than after 20 windows of sampling.
w2, sw2, sd2 = s.run_model(R0=2.0, inf_t0=100.0, Delta=15, delta_t=45, seed=sd)
assert sd2 == sd and (w2 == w).all() and (sw2 == sw).all(), (
    "re-running with the stored seed did NOT reproduce the trajectory")
print("seed      : %d reproduces its own trajectory exactly OK" % sd)
print("pre-flight OK")
PYCHECK
set +e

echo "started   : $(date)"
echo

python abc_seir_smc.py
STATUS=$?

# If the run died for any reason SLURM did not kill outright -- a pyabc error, a
# bad particle, an out-of-memory in one worker -- pull whatever generations did
# finish out of history.db rather than leaving the job with nothing to show.
# (A SIGKILL at the wall-time limit stops the shell too, so this cannot run then;
# that is what `python export_smc.py` afterwards is for.)
if [ "$STATUS" -ne 0 ]; then
    echo
    echo "abc_seir_smc.py exited with status $STATUS -- exporting whatever finished"
    python export_smc.py || echo "export_smc.py also failed; history.db is still there"
fi

echo
echo "finished  : $(date)   (exit $STATUS)"
OUT="../output/smc_B/$SMC_RUNTAG/n$SMC_NHIST/w$SMC_WEEK"
echo "-> $OUT/posterior.csv"
echo "-> $OUT/trajectories.npz"
echo "-> $OUT/S_weekly.npz"
echo "-> $OUT/Rt.npz"
echo "-> $OUT/populations.csv"
echo "-> $OUT/history.db"
echo
echo "when every array task is done, copy ../output/smc_B/$SMC_RUNTAG/ down whole"
exit $STATUS
