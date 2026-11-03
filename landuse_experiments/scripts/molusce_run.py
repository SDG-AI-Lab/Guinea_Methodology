"""MOLUSCE run from code, without QGIS's window (grid_100m notebook 07): the plugin's own algorithms
(MOLUSCE 5.1.0, plugins/molusce/algorithms), called in the order its dialog calls them, with the
settings of our GUI runs (QGIS's saved MOLUSCE settings: random sampling, 50,000 samples,
neighbourhood 1; ANN: 10 hidden nodes, 1,000 iterations, learning rate 0.1, momentum 0.05; LR: 100
iterations, the dialog's default).

  1. cross-table and change map of the initial (2023) and final (2024) maps   (Area changes tab)
  2. train the model on the 2023 map and the factors dated 2023               (Transition potential tab)
  3. one simulation step from the 2024 map with the factors dated 2024        (Simulation tab, "separate
     spatial variables version for simulation"), with transition potentials

It writes what the dialog writes: the simulated map, the certainty map and one transition potential
map per class (potential_<method>__from_<i>_0_to_<j>_0.tif). Runs in QGIS's Python:

    "C:\\Program Files\\QGIS 3.22.9\\bin\\python-qgis-ltr.bat" molusce_run.py <benchmark folder> <lr|ann> <out folder> [seed] [--float32]

MOLUSCE draws its samples and starting weights with numpy's global random generator, unseeded in the
GUI; the seed here makes a run repeatable.

float32 (option, needed at 100 m): MOLUSCE holds every factor map twice in float64 (the raster as read,
dataprovider.Raster._read, and a copy in the sampler, sampler.Sampler.__init__): about 11 GB for the 12
factors at 100 m. With this option those two arrays are float32, nothing else changes (the 50,000 training
samples and the models stay float64). Notebook 07 checks at 250 m that it gives the same result. In
every run the factor rasters are released once the sampler has copied them (release_factor_maps).
"""
import ctypes
import gc
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path.home() / "AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins"))
import numpy as np                                                          # noqa: E402
from molusce.algorithms.dataprovider import Raster                          # noqa: E402
from molusce.algorithms.models.area_analysis.manager import AreaAnalyst     # noqa: E402
from molusce.algorithms.models.crosstabs.manager import CrossTableManager   # noqa: E402
from molusce.algorithms.models.lr.lr import LR                              # noqa: E402
from molusce.algorithms.models.mlp.manager import MlpManager                # noqa: E402
from molusce.algorithms.models.simulator.sim import Simulator              # noqa: E402

SETTINGS = {"samplingMode": "Random", "samples": 50_000, "neighbourhood": 1,
            "ann": {"topology": [10], "epochs": 1000, "valPercent": 20, "lrate": 0.1, "momentum": 0.05},
            "lr": {"maxiter": 100}}


class _Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_gb():
    """Peak working set of this process so far (Windows)."""
    c = _Counters()
    c.cb = ctypes.sizeof(c)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    info(current(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 1e9


class _Float32:
    """numpy for MOLUSCE's dataprovider and sampler modules, except that the two full-map float64
    arrays they allocate (np.ma.zeros / np.ma.array with dtype=float) are float32."""

    class _Ma:
        def __getattr__(self, name):
            return getattr(np.ma, name)

        @staticmethod
        def _dtype(kw):
            if kw.get("dtype") is float:
                kw["dtype"] = np.float32
            return kw

        def zeros(self, *a, **kw):
            return np.ma.zeros(*a, **self._dtype(kw))

        def array(self, *a, **kw):
            return np.ma.array(*a, **self._dtype(kw))

    ma = _Ma()

    def __getattr__(self, name):
        return getattr(np, name)


def release_factor_maps():
    """MOLUSCE's sampler copies the factor maps when it is built (Sampler.__init__); training samples
    and predictions then read only that copy (Sampler.get_factors). The factor rasters' own arrays are
    dropped at that point, so the maps are held once instead of twice. No value changes."""
    from molusce.algorithms.models.sampler import sampler
    build = sampler.Sampler.__init__

    def init(self, state, factors, output=None, ns=0):
        build(self, state, factors, output, ns)
        for raster in factors:
            raster.bands = None
        gc.collect()

    sampler.Sampler.__init__ = init


def use_float32():
    from molusce.algorithms import dataprovider
    from molusce.algorithms.models.sampler import sampler
    dataprovider.np = sampler.np = _Float32()


def factors(folder):
    return [Raster(str(p)) for p in sorted(Path(folder).glob("*.tif"))]


def main(bench, method, out, seed=0, float32=False):
    bench, out = Path(bench), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    release_factor_maps()
    if float32:
        use_float32()
    np.random.seed(seed)
    log, t0 = {"method": method, "seed": seed, "float32": float32, "settings": SETTINGS, "stages": []}, time.time()

    def stage(name):
        log["stages"].append({"stage": name, "minutes": (time.time() - t0) / 60, "peak_GB": peak_gb()})
        print(f"{name}: {(time.time() - t0) / 60:.1f} min, peak {peak_gb():.2f} GB", flush=True)

    initial = Raster(str(bench / "inputs" / "landcover_2023.tif"))
    final = Raster(str(bench / "inputs" / "landcover_2024.tif"))
    crosstab = CrossTableManager(initial, final)
    crosstab.computeCrosstable()
    change_map = AreaAnalyst(initial, final).getChangeMap()
    stage("Area changes")

    train_factors = factors(bench / "inputs" / "factors_2023")
    ns = SETTINGS["neighbourhood"]
    if method == "lr":
        model = LR(ns=ns)
        model.setMaxIter(SETTINGS["lr"]["maxiter"])
        model.setState(initial)
        model.setFactors(train_factors)
        model.setOutput(change_map)
        model.setMode(SETTINGS["samplingMode"])
        model.setSamples(SETTINGS["samples"])
        model.startTrain()
        log["training"] = {"kappa": float(model.getKappa()), "pseudo_R2": float(model.getPseudoR())}
    else:
        a = SETTINGS["ann"]
        model = MlpManager(ns=ns)
        model.createMlp(initial, train_factors, change_map, a["topology"])
        model.setTrainingData(initial, train_factors, change_map, mode=SETTINGS["samplingMode"], samples=SETTINGS["samples"])
        model.setEpochs(a["epochs"])
        model.setValPercent(a["valPercent"])
        model.setLRate(a["lrate"])
        model.setMomentum(a["momentum"])
        model.setContinueTrain()
        model.startTrain()
        log["training"] = {"train_error": float(model.getTrainError()), "min_validation_error": float(model.getMinValError()),
                           "kappa": float(model.getKappa())}
    stage("Training")
    for attr in ("factors", "sampler", "state", "output"):                  # the simulation reads the 2024 factors
        if hasattr(model, attr):
            setattr(model, attr, None)
    del train_factors, initial, change_map
    gc.collect()

    sim = Simulator(final, factors(bench / "inputs" / "factors_2024"), model, crosstab)
    sim.setIterationCount(1)
    sim.setCalcTransitions(True)
    sim.simN()
    stage("Simulation")

    sim.getState().save(str(out / f"molusce_{method}_2025.tif"), nodata=0)
    sim.getConfidence().save(str(out / f"{method}_certainty_function.tif"), nodata=255)
    analyst = AreaAnalyst(final, second=None)
    for k, raster in sim.getTransitionPotentials().items():
        i, j = (str(c).replace(".", "_") for c in analyst.decode(int(k)))
        raster.save(str(out / f"potential_{method}__from_{i}_to_{j}.tif"))
    stage("Written")
    (out / f"molusce_{method}_run.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--float32"]
    main(args[0], args[1], args[2], int(args[3]) if len(args) > 3 else 0, float32="--float32" in sys.argv)
