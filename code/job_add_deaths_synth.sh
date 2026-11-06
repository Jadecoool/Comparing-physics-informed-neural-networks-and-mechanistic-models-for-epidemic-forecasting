#!/bin/bash
#SBATCH --job-name=synth_age
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=computeshort
#SBATCH --array=3-30

# Replay the accepted particles of a synthetic ABC-SMC run and save the
# deaths-by-age that the fit did not store. One array task per window.
# No calibration is redone: posterior.csv + seeds.npz go back in, and the
# weekly new infections are checked against trajectories.npz.
#
#   SYNTH_SCEN=R0_2.0_noise_light sbatch job_add_deaths_synth.sh
#   SYNTH_SCEN=R0_1.5_noise_light sbatch job_add_deaths_synth.sh
#   SYNTH_SCEN=R0_1.5_noise_mild  sbatch job_add_deaths_synth.sh
#   SYNTH_SCEN=R0_2.0_noise_mild  sbatch job_add_deaths_synth.sh
#
# 1000 particles per window, about 0.5 s each, so roughly 10-20 min per task.
# Writes deaths_by_age.npz next to the other files in
# ../output/smc_synth/<scenario>/nall/w<t>/

cd "$SLURM_SUBMIT_DIR"

module load python/3.8.5-gcc-12.2.0
source /data/home/ahw917/project_1_for_apocrita/proj1/bin/activate

export SMC_WEEK="${SMC_WEEK:-${SLURM_ARRAY_TASK_ID:-12}}"
export SYNTH_SCEN="${SYNTH_SCEN:-R0_2.0_noise_light}"
export SMC_NHIST="${SMC_NHIST:-all}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "scenario : $SYNTH_SCEN   n_history: $SMC_NHIST   window: $SMC_WEEK"
echo "started  : $(date)"
echo

python add_deaths_by_age_synth.py

echo
echo "finished : $(date)"
