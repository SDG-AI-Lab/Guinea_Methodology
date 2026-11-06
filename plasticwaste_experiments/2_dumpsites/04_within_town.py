"""Same-town labels: cross-validated inside each blind grid, holding out 400 m blocks of
ground. Answers whether local labels help, which the other-town test cannot."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
B = importlib.import_module("02_benchmark")


def main():
    rows = []
    for t in ("conakry", "boke"):
        m = B.meta(t)
        y, g = m.y.to_numpy(), m.cell.to_numpy()
        for rep, name in B.REPS.items():
            X = B.load(t, rep)
            s = np.zeros(len(y))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
                c = B.choose_c(X[tr], y[tr], g[tr])
                s[te] = B.pipe(c).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            lo, hi = B.ci(y, s, g)
            rows.append({"test": t, "rep": name, "auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
                         "auc_p05": lo, "auc_p95": hi, "blocks": len(set(g))})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(B.OUT / "04_within_town.csv", index=False)


if __name__ == "__main__":
    main()
