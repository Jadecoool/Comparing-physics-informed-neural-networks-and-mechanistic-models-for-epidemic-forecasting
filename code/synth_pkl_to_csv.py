"""
Turn the four synthetic .pkl scenarios into .csv, once.

    python synth_pkl_to_csv.py

The pickles were written with numpy 2.x, which renamed numpy.core to
numpy._core, so older numpy cannot unpickle them. The lines below point the new
names at the old module. After this nothing else has to deal with it.
"""

import os
import pickle
import sys

import numpy as np

for new in ["numpy._core", "numpy._core.multiarray", "numpy._core.numeric",
            "numpy._core._multiarray_umath"]:
    old = new.replace("_core", "core")
    try:
        __import__(old)
        sys.modules[new] = sys.modules[old]
    except Exception:
        pass


here = os.path.dirname(os.path.abspath(__file__))
folder = here + "/../data/synthetic"

SCENARIOS = ["R0_1.5_noise_light", "R0_1.5_noise_mild",
             "R0_2.0_noise_light", "R0_2.0_noise_mild"]

for name in SCENARIOS:
    obj = pickle.load(open(folder + "/synthetic_" + name + ".pkl", "rb"))
    day = np.asarray(obj["t"], dtype=float)
    inc = np.asarray(obj["incidence"], dtype=float)

    out = folder + "/" + name + ".csv"
    fh = open(out, "w", newline="\n")
    fh.write("week,day,incidence\n")
    for j in range(len(inc)):
        fh.write("%d,%d,%d\n" % (j, int(day[j]), int(inc[j])))
    fh.close()

    print("%-22s %2d weeks  peak %9.0f at week %2d  -> %s"
          % (name, len(inc), inc.max(), int(inc.argmax()),
             os.path.basename(out)))
