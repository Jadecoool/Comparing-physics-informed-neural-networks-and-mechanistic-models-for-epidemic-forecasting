# ABC-SMC calibration and rolling forecasts of an age-structured SEIR model

Code and results for an age-structured stochastic SEIR model,
calibrated with ABC-SMC (pyabc), on synthetic data and on London COVID-19 deaths.

## Repository layout

```
code/      model, ABC-SMC and SLURM job scripts
data/
  regions/London/      contact matrices, population, IFR, weekly deaths
  synthetic/           the 4 synthetic scenarios (.pkl originals and .csv)
output/
  smc_synth/                        experiment 1
  smc_synth_retro/                  experiment 2
  smc_B/seir_logwmape_skipzero/     experiment 3
```

## Requirements

Python 3.8 with `numpy`, `pandas`, `scipy`, `pyabc`, `numba`, `joblib`, `pyyaml`
(`requirements.txt`). The drivers must run on Linux or macOS.

## Main scripts

| Experiment | Main script | Job script |
|---|---|---|
| 1. Synthetic rolling forecasts | `abc_smc_synth.py` | `job_abc_smc_synth.sh` |
| 2. Synthetic retrospective fitting | `abc_smc_synth_retro.py` | `job_abc_smc_synth_retro.sh` |
| 3. Real-data rolling forecasts | `abc_seir_smc.py` | `job_abc_seir_smc.sh` |
