"""Fit forestatrisk's binomial iCAR model at 100 m (runs in .venv_tools). Usage: work_folder

forestatrisk's binomial iCAR model: logistic regression of loss on the scaled predictors, plus a
spatial random effect for each 10 km cell with an intrinsic CAR prior over its eight neighbours,
fitted by MCMC (forestatrisk.model_binomial_iCAR). The forest cells to predict are read from
X_fit.npy and X_test.npy a chunk at a time (memory-mapped), and the predictions are written to
far_fit.npy and far_test.npy.
"""
import os
import sys

import numpy as np
import pandas as pd
import forestatrisk as far

work = sys.argv[1]
data = np.load(os.path.join(work, "training.npz"))
names = [f"x{i}" for i in range(len(data["names"]))]       # patsy-safe column names
nx, ny = int(data["ncell_x"]), int(data["ncell_y"])

n_neighbors, neighbors = [], []
for cy in range(ny):
    for cx in range(nx):
        adj = [(cy + dy) * nx + (cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and 0 <= cy + dy < ny and 0 <= cx + dx < nx]
        n_neighbors.append(len(adj))
        neighbors.extend(adj)

frame = pd.DataFrame(data["X"], columns=names)
frame["fcc23"] = 1 - data["y"]                                # forestatrisk: 1 = forest remains
frame["trial"] = 1
frame["cell"] = data["cell"]
formula = "I(1 - fcc23) + trial ~ " + " + ".join(f"scale({n})" for n in names) + " + cell"
model = far.model_binomial_iCAR(suitability_formula=formula, data=frame, n_neighbors=np.array(n_neighbors),
                                neighbors=np.array(neighbors), burnin=1000, mcmc=1000, thin=1, beta_start=-99)
print(model)


def predict(X, cells, chunk=500_000):
    out = np.full(len(X), np.nan, dtype="float32")
    for i in range(0, len(X), chunk):
        block = np.asarray(X[i:i + chunk])
        ok = np.isfinite(block).all(axis=1)
        new = pd.DataFrame(block[ok], columns=names)
        new["cell"] = cells[i:i + chunk][ok]
        part = out[i:i + chunk]
        part[ok] = model.predict(new_data=new)
    return out


for which in ("fit", "test"):
    X = np.load(os.path.join(work, f"X_{which}.npy"), mmap_mode="r")
    cells = np.load(os.path.join(work, f"cell_{which}.npy"))
    np.save(os.path.join(work, f"far_{which}.npy"), predict(X, cells))
    print(f"forestatrisk: {which} predictions saved ({len(X):,} cells)", flush=True)
