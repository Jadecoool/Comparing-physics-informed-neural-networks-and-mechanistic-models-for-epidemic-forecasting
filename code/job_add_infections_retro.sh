#!/bin/bash
#SBATCH --job-name=retro_infage
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=computeshort
#SBATCH --array=0-3

# Replay the accepted particles of the retrospective synthetic fits and save
# the infections-by-age that the fit did not store. ONE TASK PER SCENARIO, so
# the array index picks the scenario, not a window.
#
#   sbatch job_add_infections_retro.sh              # all four, in parallel
#   sbatch --array=2 job_add_infections_retro.sh    # just R0_2.0_noise_light
#
# No calibration is redone: posterior.csv + seeds.npz go back in, and the 16
# age rows are checked against trajectories.npz before anything is written.
#
# 1000 particles per scenario, about 0.5 s each, so roughly 10-20 min per task.
# Writes infections_by_age.npz into ../output/smc_synth_retro/<scenario>/
#
# NOTE sbatch exports your login shell's environment. If SYNTH_SCEN is set
# there, all four tasks take that one scenario. Run `unset SYNTH_SCEN` first.

cd "$SLURM_SUBMIT_DIR"

module load python/3.8.5-gcc-12.2.0
source /data/home/ahw917/project_1_for_apocrita/proj1/bin/activate

SCENARIOS=(R0_1.5_noise_light R0_1.5_noise_mild R0_2.0_noise_light R0_2.0_noise_mild)

export SYNTH_SCEN="${SYNTH_SCEN:-${SCENARIOS[${SLURM_ARRAY_TASK_ID:-0}]}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "scenario : $SYNTH_SCEN"
echo "started  : $(date)"
echo

python add_infections_by_age_retro.py

echo
echo "finished : $(date)"
echo "-> ../output/smc_synth_retro/$SYNTH_SCEN/infections_by_age.npz"
