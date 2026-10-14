"""
Post-copy verification for `plasticwaste_tool`.

Answers one question: "did this copy of the module land correctly, and does it
run?" — offline, with no credentials, no network and no billable requests.
Run it immediately after copying the module into a repository or a container,
before wiring anything to it.

    python -m plasticwaste_tool.verify_install
    python -m plasticwaste_tool.verify_install --data     # also check run outputs

WHAT IT CHECKS, AND WHY EACH ONE EARNS ITS PLACE
------------------------------------------------
  1. torch loads first. On Windows, torch and the GEE/GDAL stack both ship
     native DLLs and whichever initialises first wins; the loser dies with
     WinError 1114, which looks like a broken torch install rather than an
     ordering problem. This check runs first precisely so the rest of the
     script is meaningful.
  2. Every module imports. A missing file or a broken relative import after a
     partial copy shows up here rather than three weeks later in a Celery task.
  3. The frozen heads load and their metadata parses. These are binary files
     that a blanket `*.joblib` gitignore rule silently drops.
  4. The backend interface returns the documented schema while disabled. This
     is what the platform calls; it must be wireable before the model is on.
  5. The FDI chain computes real numbers on a synthetic scene.
  6. Feature extraction produces a 2048-d vector and the head scores it. This
     is the one check that exercises torch, torchvision weights and the frozen
     probe together — i.e. everything a real scan does except fetching tiles.

Exit code 0 if everything required passed, 1 otherwise. Data checks (--data)
are reported but never fail the run: the run outputs are large and a code-only
deployment is a legitimate configuration.
"""

from __future__ import annotations

# MUST be first. See check 1 and the note in dumpsite_tool/README.md.
try:
    import torch  # noqa: F401
    _TORCH_ERR = None
except Exception as _exc:  # ImportError, or OSError 1114 on Windows
    _TORCH_ERR = _exc

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The modules under test import each other as `plasticwaste_tool.<x>`, so the
# PARENT of this file has to be importable. Running `python verify_install.py`
# puts this directory on sys.path, not its parent, and every check then fails
# with "No module named 'plasticwaste_tool'" -- which reads like a broken
# install rather than a broken invocation. Inserting the parent makes the
# script work the same way from anywhere:
#     python verify_install.py
#     python plasticwaste_tool/verify_install.py
#     python -m plasticwaste_tool.verify_install
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

_PASS, _FAIL, _WARN = [], [], []


def check(name, fn, required=True):
    """Run one check. `fn` returns a detail string, or raises."""
    try:
        detail = fn() or ""
    except Exception as exc:
        (_FAIL if required else _WARN).append(name)
        mark = "FAIL" if required else "warn"
        print(f"  [{mark}] {name}\n         {type(exc).__name__}: {exc}")
        if required and "-v" in sys.argv:
            traceback.print_exc()
        return False
    _PASS.append(name)
    print(f"  [ ok ] {name}" + (f"  — {detail}" if detail else ""))
    return True


# ---------------------------------------------------------------- 1. torch
def _torch_first():
    if _TORCH_ERR is not None:
        raise RuntimeError(
            f"{type(_TORCH_ERR).__name__}: {_TORCH_ERR}\n"
            "         On Windows this means the GEE/GDAL stack initialised "
            "first.\n"
            "         Put `import torch` above every plasticwaste_tool import "
            "in the entry point.")
    return f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}"


# -------------------------------------------------------------- 2. imports
CORE = ["config", "api_connector", "data_loader", "validation", "stats",
        "gee_auth", "geo_env", "detection", "spectral", "pipeline",
        "deployment"]
FDI = ["fdi_tool", "fdi_tool.spectral", "fdi_tool.detection",
       "fdi_tool.pipeline", "fdi_tool.deployment"]
DUMP = ["dumpsite_tool", "dumpsite_tool.scan", "dumpsite_tool.inference",
        "dumpsite_tool.features", "dumpsite_tool.point_model",
        "dumpsite_tool.footprint", "dumpsite_tool.train_head"]
# feature_probe is deliberately NOT in DUMP: it imports the AerialWaste
# architecture, a 230 MB directory excluded from the repository because the
# module ships ImageNet features instead. Nothing on the deployment path
# touches it — see check 6, which exercises the path that matters.
CHAR = ["characterisation_tool", "characterisation_tool.separability",
        "characterisation_tool.mobilisation", "characterisation_tool.extent",
        "characterisation_tool.fdi_on_land", "characterisation_tool.s2_indices",
        "characterisation_tool.evaluate_indices",
        "characterisation_tool.hyperspectral_coverage",
        "characterisation_tool.enmap_spectra",
        "characterisation_tool.enmap_download",
        "characterisation_tool.extent_annotator",
        "characterisation_tool.ingest_annotations",
        "characterisation_tool.extent_second_read"]


def _import_group(names):
    def run():
        bad = []
        for n in names:
            try:
                importlib.import_module("plasticwaste_tool." + n)
            except Exception as exc:
                bad.append(f"{n} ({type(exc).__name__}: {str(exc)[:60]})")
        if bad:
            raise ImportError("; ".join(bad))
        return f"{len(names)} modules"
    return run


# --------------------------------------------------------- 3. frozen heads
def _frozen_heads():
    from .dumpsite_tool import scan
    models = HERE / "dumpsite_tool" / "models"
    out = []
    for name in ("probe.joblib", "probe_screen.joblib"):
        if not (models / name).exists():
            raise FileNotFoundError(
                f"{name} is missing. It is ~67 KB and IS meant to be "
                "committed — check that a blanket '*.joblib' gitignore rule "
                "has not caught it.")
        _est, _ = scan.load_head(name)
        meta = json.loads((models / name.replace(".joblib", "_meta.json"))
                          .read_text(encoding="utf-8"))
        out.append(f"{name} [{meta.get('backbone', '?')}]")
    tiers, default = scan.read_tiers()
    return f"{', '.join(out)}; tiers {tiers}, default '{default}'"


# ----------------------------------------------------- 4. backend interface
def _interface():
    from . import detection as D
    g = D.classify_dumpsite_candidates()
    if list(g.columns) != D.DUMPSITE_SCHEMA:
        raise AssertionError(
            f"schema drift: {list(g.columns)} != {D.DUMPSITE_SCHEMA}")
    if not g.empty:
        raise AssertionError("expected an empty layer while the model is "
                             "disabled")
    return f"{len(D.DUMPSITE_SCHEMA)} columns, empty while disabled"


# ------------------------------------------------------------- 5. FDI chain
def _fdi_chain():
    from .geo_env import fix_proj
    fix_proj()
    import numpy as np
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from .fdi_tool import pipeline as P

    H = W = 200
    bands = P._synthetic_scene(H, W)
    profile = {"crs": CRS.from_epsg(4326), "width": W, "height": H,
               "transform": from_bounds(-13.90, 9.30, -13.85, 9.35, W, H)}
    res = P.run_from_arrays(bands, profile, acquisition_date="2024-01-15")
    n = res["summary"]["n_patches"]
    if n < 1:
        raise AssertionError("no synthetic debris patch detected")
    return (f"{n} patches, threshold "
            f"{res['threshold_sensitivity']['base_threshold_fdi']:.5f} "
            f"({res['threshold_sensitivity']['rule']})")


# ------------------------------------------- 6. features + head, end to end
def _scoring_chain():
    """
    The check that matters most: a real image through the real feature
    extractor into the real frozen head. Everything a national scan does
    except fetching tiles over the network.
    """
    import numpy as np
    from PIL import Image
    import tempfile
    from .dumpsite_tool import features as bm, scan

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tile.png"
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (800, 800, 3), dtype=np.uint8)
        Image.fromarray(noise).save(p)
        X = bm.imagenet_features([str(p)])
        if X.shape != (1, 2048):
            raise AssertionError(f"expected (1, 2048) features, got {X.shape}")
        # load_head returns (estimator, metadata); _score wants the
        # estimator, which is the same unpacking scan.scan() itself does.
        model, meta = scan.load_head("probe.joblib")
        s = scan._score([str(p)], model)
    lo, hi = float(np.min(s)), float(np.max(s))
    if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
        raise AssertionError(f"score out of [0,1]: {lo}, {hi}")
    return (f"2048-d features -> head score {lo:.3f} "
            f"(random-noise tile, {meta.get('backbone', '?')})")


# ------------------------------------------------------------ data (opt-in)
DATA = [
    ("runs/hyperspectral_coverage/coverage_summary.json", "gate 1 + methodology"),
    ("runs/s2_plastic_indices/evaluation.json", "gate 2 + methodology"),
    ("runs/fdi_on_land/fdi_land_evaluation.json", "gate 3 + methodology"),
    ("runs/dumpsite_extent/extent_annotated_summary.json", "extent + methodology"),
    ("runs/dumpsite_extent/extent_agreement.json", "second read + methodology"),
    ("runs/enmap_spectra/separability.json", "gate 4 + methodology"),
    ("runs/mobilisation/mobilisation_summary.json", "mobilisation + methodology"),
    ("runs/figures", "characterisation figures"),
    ("runs/figures_doc", "archived marine figures"),
    ("manual_qa/regions/national_random_sample.geojson", "the label set"),
    ("manual_qa/regions/conakry_blindgrid_review.geojson", "blind-grid labels"),
    ("manual_qa/_metrics_pt80_google.json", "imagery comparison"),
    ("manual_qa/_metrics_pt80_esri.json", "imagery comparison"),
]


def report_data():
    print("\nData artefacts (optional — a code-only deployment is valid):")
    missing = 0
    for rel, why in DATA:
        p = HERE / rel
        if p.exists():
            print(f"  [ ok ] {rel}")
        else:
            missing += 1
            print(f"  [ -- ] {rel}   ({why})")
    if missing:
        print(f"\n  {missing} of {len(DATA)} absent. Consequences:")
        print("    • the characterisation notebook cannot run")
        print("    • 25_build_module_methodology.py cannot regenerate the doc")
        print("    • the FDI and dumpsite code are UNAFFECTED — they need none "
              "of this")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", action="store_true",
                    help="also report on run outputs and label files")
    args = ap.parse_args()

    print("=" * 68)
    print(" plasticwaste_tool — install verification")
    print("=" * 68)
    print(f" location: {HERE}")
    print(f" python  : {sys.version.split()[0]}\n")

    check("torch initialised before the GDAL stack", _torch_first)
    check("shared modules import", _import_group(CORE))
    check("fdi_tool imports", _import_group(FDI))
    check("dumpsite_tool imports", _import_group(DUMP))
    check("characterisation_tool imports", _import_group(CHAR))
    check("frozen heads load", _frozen_heads)
    check("backend interface returns the documented schema", _interface)
    check("FDI detection chain runs on a synthetic scene", _fdi_chain)
    check("features + frozen head score an image", _scoring_chain)

    if args.data:
        report_data()

    print("\n" + "=" * 68)
    print(f" {len(_PASS)} passed, {len(_FAIL)} failed"
          + (f", {len(_WARN)} warnings" if _WARN else ""))
    if _FAIL:
        print(" FAILED: " + ", ".join(_FAIL))
    print("=" * 68)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    # Several checks import modules that use package-relative imports
    # (`from . import config`). Those only resolve when this file is itself
    # running as part of the package -- as a plain script, `__package__` is
    # empty and they fail with "attempted relative import with no known parent
    # package", which looks like a broken install and is not one.
    #
    # Rather than document a required invocation that people will get wrong,
    # re-execute as a module when we were not started as one. Guarded by an
    # environment variable so a failure in the re-exec cannot recurse.
    import os

    if not __package__ and not os.environ.get("_PW_VERIFY_REEXEC"):
        os.environ["_PW_VERIFY_REEXEC"] = "1"
        import subprocess

        sys.exit(subprocess.call(
            [sys.executable, "-m", "plasticwaste_tool.verify_install",
             *sys.argv[1:]],
            cwd=str(HERE.parent)))

    sys.exit(main())
