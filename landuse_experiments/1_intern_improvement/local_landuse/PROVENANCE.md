# Provenance: the intern's `local_landuse` package

Verbatim copy. **Do not edit.** Where a notebook needs different behaviour, the
change is made from `nb_helpers.py` at run time and explained in the notebook.

**One line changed (2026-10-24), a file location only:** the package was moved from
`landuse_experiments/` into `landuse_experiments/1_intern_improvement/`, so `DEFAULT_DATA`
in `comparison.py` now looks two folders up (`parents[2]`) instead of one (`parents[1]`), to
find the same `landuse_experiments/data/`. No method, setting or result changes.

| | |
|---|---|
| Source repo | `SDG-AI-Lab/Intern_Tasks`, branch `guinea-project` |
| Commit | `d4a3c8e70bcf4b4ce9a393dbff023c0e3d658e2c` |
| Source path | `guinea_project/Land Use-Deforestation/local_landuse/` |
| Copied | 2026-10-18 |

Also copied, from the same commit:

- `tests/`: the intern's 74 tests (synthetic fixtures only)
- `intern_docs/`: the six experiment write-ups (`FOM40_CEILING_ANALYSIS.md` is the summary)
- `requirements-intern.txt`: the intern's pins (Python 3.14). This folder's
  `../requirements.txt` is the environment actually used here (Python 3.10).

Not copied: the intern's notebooks (0.3–9 MB each, outputs embedded) and their
older copies of `landuse_tool/` and the other tools. Their `outputs/` folder was
never shared; notebooks 01–04 regenerate it.

The package reads `../data/` and writes `../outputs/`, which is the layout of
`landuse_experiments/`.

Verified 2026-10-18: `python -m pytest local_landuse/tests -q` → 74 passed
(Python 3.10.0, numpy 2.2.6, scikit-learn 1.7.2, xgboost 3.2.0, rasterio 1.4.4).
