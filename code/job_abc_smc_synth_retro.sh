#!/bin/bash
#SBATCH --job-name=smc_retro
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=0-3

# Retrospective ABC-SMC on the synthetic scenarios. ONE FIT PER SCENARIO, no
# rolling windows, so the array index picks the scenario, not the window.
#
#   sbatch job_abc_smc_synth_retro.sh              # all four scenarios
#   sbatch --array=2 job_abc_smc_synth_retro.sh    # just R0_2.0_noise_light
#
# Or ignore the array and name the scenario:
#
#   SYNTH_SCEN=R0_2.0_noise_mild sbatch --array=0 job_abc_smc_synth_retro.sh
#
# How many weeks are fitted is decided inside abc_smc_synth_retro.py by the
# scenario name: R0 = 1.5 uses all 35, R0 = 2.0 uses the first 25.
#
# Needs ../data/synthetic/<scenario>.csv from synth_pkl_to_csv.py.
# SMC_HOURS must stay below --time so pyabc stops itself and saves.

cd "$SLURM_SUBMIT_DIR"

module load python/3.8.5-gcc-12.2.0
source /data/home/ahw917/project_1_for_apocrita/proj1/bin/activate

SCENARIOS=(R0_1.5_noise_light R0_1.5_noise_mild R0_2.0_noise_light R0_2.0_noise_mild)

export SYNTH_SCEN="${SYNTH_SCEN:-${SCENARIOS[${SLURM_ARRAY_TASK_ID:-0}]}}"
export SMC_POP="${SMC_POP:-1000}"
export SMC_GENS="${SMC_GENS:-10}"
export SMC_HOURS="${SMC_HOURS:-11}"
export SMC_NPROCS="${SLURM_CPUS_PER_TASK:-16}"

# one thread per worker; pyabc forks and SEIR seeds the global numpy state
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

echo "scenario : $SYNTH_SCEN"
echo "started  : $(date)"
echo

python abc_smc_synth_retro.py

echo
echo "finished : $(date)"
echo "-> ../output/smc_synth_retro/$SYNTH_SCEN"
