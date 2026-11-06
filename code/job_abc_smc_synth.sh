#!/bin/bash
#SBATCH --job-name=smc_synth
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --array=3-30

# ABC-SMC on the synthetic scenarios, one array task per window.
# 35 weeks and horizon 4, so the windows are 3..30.
#
#   SYNTH_SCEN=R0_2.0_noise_light SMC_NHIST=all sbatch job_abc_smc_synth.sh
#   SYNTH_SCEN=R0_2.0_noise_light SMC_NHIST=4   sbatch job_abc_smc_synth.sh
#
# Scenarios: R0_1.5_noise_light, R0_1.5_noise_mild,
#            R0_2.0_noise_light, R0_2.0_noise_mild
#
# Needs ../data/synthetic/<scenario>.csv from synth_pkl_to_csv.py.
# SMC_HOURS must stay below --time so pyabc stops itself and saves.

cd "$SLURM_SUBMIT_DIR"

module load python/3.8.5-gcc-12.2.0
source /data/home/ahw917/project_1_for_apocrita/proj1/bin/activate

export SMC_WEEK="${SMC_WEEK:-${SLURM_ARRAY_TASK_ID:-12}}"
export SYNTH_SCEN="${SYNTH_SCEN:-R0_2.0_noise_light}"
export SMC_NHIST="${SMC_NHIST:-all}"
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

echo "scenario : $SYNTH_SCEN   n_history: $SMC_NHIST   window: $SMC_WEEK"
echo "started  : $(date)"
echo

python abc_smc_synth.py

echo
echo "finished : $(date)"
echo "-> ../output/smc_synth/$SYNTH_SCEN/n$SMC_NHIST/w$SMC_WEEK"
