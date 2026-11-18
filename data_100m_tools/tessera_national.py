"""TESSERA embeddings over every forest cell of Guinea, reduced to 16 numbers.

The deforestation test (deforestation_experiments notebook 11) found the improved model gains
+0.11 to +0.15 AUC with TESSERA in 16 forest blocks, and that its first 16 principal components
score as well as all 128 numbers. Verra's tests need the whole country, so this reads TESSERA
over the 2,048 tiles holding forest and keeps, for each of the 12.6 million cells with forest in
2000 (model.hansen() order), the 16 components: 0.4 GB a year instead of 300-370 GB.

  index   (landuse_experiments/.venv)  python data_100m_tools/tessera_national.py index
          the tiles and their forest cells, and the 16-component rotation fitted once on the
          forest cells of the 24 sample blocks, all years 2017-2025 (fixed, as the platform would)
  fetch   (venv_tessera, Python >= 3.12) python data_100m_tools/tessera_national.py fetch YEAR --workers 4
          [--limit N] reads each tile, averages the 10 x 10 pixels of each cell (tessera_fetch.py),
          projects to the 16 components and writes them into STORE/tessera/national/pc16_<YEAR>.npy
          (float16 memmap, cells x 16, NaN where no data). Finished tiles are listed in
          done_<YEAR>.txt, so a stopped run restarts where it was.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

STORE = Path(os.environ.get("GUINEA_DATA_100M", r"E:\SDG AI Lab\Guinea\data_100m"))
TES = STORE / "tessera"
NAT = TES / "national"
N_PC = 16


def index():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import grid100 as G
    NAT.mkdir(parents=True, exist_ok=True)
    forest = G.read_layer("hansen_shares", ["forest2000"])["forest2000"]
    keep = np.flatnonzero(forest > 0)                          # = model.hansen()["idx"]
    lon, lat = G.lon_lat(G.cells()[keep])
    tx, ty = np.floor(lon * 10).astype(int), np.floor(lat * 10).astype(int)
    keys, inv = np.unique(np.stack([tx, ty], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(keys)))
    np.savez(NAT / "index.npz", order=order.astype("int32"), starts=starts.astype("int64"),
             keys=keys.astype("int32"), lon=lon.astype("float64"), lat=lat.astype("float64"))

    # 16-component rotation from the forest cells of the 24 blocks, all years
    meta = json.loads((TES / "blocks.json").read_text())["blocks"]
    cells = np.load(TES / "blocks_cells.npz")
    X = []
    for b in meta:
        pos = cells[f"{b['name']}_pos"]
        f = forest[pos] > 0
        for y in range(2017, 2026):
            E = np.load(TES / str(y) / f"{b['name']}.npy").astype(np.float32)[f]
            X.append(E[np.isfinite(E).all(1)])
    X = np.concatenate(X)
    X = X[np.random.default_rng(0).choice(len(X), min(400_000, len(X)), replace=False)]
    mean = X.mean(0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    var = (s ** 2) / (s ** 2).sum()
    np.savez(NAT / "pca16.npz", mean=mean, components=vt[:N_PC], explained=var[:N_PC])
    print(f"{len(keys)} tiles, {keep.size:,} forest cells; 16 components keep {var[:N_PC].sum():.1%} of the variance")


def _work(args):
    year, tiles, worker = args
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tessera_fetch import cell_means, PAD, ZONE_EDGE
    from geotessera import GeoTesseraZarr
    gt = GeoTesseraZarr()
    ix = np.load(NAT / "index.npz")
    pca = np.load(NAT / "pca16.npz")
    n = ix["lon"].size
    out = np.lib.format.open_memmap(NAT / f"pc16_{year}.npy", mode="r+", dtype=np.float16, shape=(n, N_PC))
    done = NAT / f"done_{year}_w{worker}.txt"
    for t in tiles:
        t0 = time.time()
        rows = ix["order"][ix["starts"][t]: ix["starts"][t + 1] if t + 1 < len(ix["starts"]) else n]
        x0, y0 = ix["keys"][t] / 10
        w, s, e, nn = x0, y0, x0 + 0.1, y0 + 0.1
        wp = max(w - PAD, ZONE_EDGE + 1e-6) if w >= ZONE_EDGE else w - PAD
        ep = min(e + PAD, ZONE_EDGE - 1e-6) if e <= ZONE_EDGE else e + PAD
        try:
            mosaic, transform, crs = gt.read_region((wp, s - PAD, ep, nn + PAD), year=year)
        except Exception as exc:                                  # a tile without data: leave NaN, note it
            with open(done, "a") as f:
                f.write(f"{t}\terror {type(exc).__name__}\n")
            continue
        v = cell_means(mosaic, transform, crs, ix["lon"][rows], ix["lat"][rows])
        ok = np.isfinite(v).all(1)
        z = np.full((rows.size, N_PC), np.nan, np.float32)
        z[ok] = (v[ok] - pca["mean"]) @ pca["components"].T
        out[rows] = z.astype(np.float16)
        out.flush()
        del mosaic
        with open(done, "a") as f:
            f.write(f"{t}\t{rows.size}\t{(~ok).mean():.4f}\t{time.time() - t0:.0f}\n")


def fetch(year, workers=4, limit=None):
    from multiprocessing import Pool
    ix = np.load(NAT / "index.npz")
    n, nt = ix["lon"].size, len(ix["keys"])
    path = NAT / f"pc16_{year}.npy"
    if not path.exists():
        m = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, N_PC)); m[:] = np.nan; m.flush(); del m
    finished = set()
    for f in NAT.glob(f"done_{year}_w*.txt"):
        finished |= {int(l.split("\t")[0]) for l in f.read_text().splitlines() if l and "error" not in l}
    todo = [t for t in range(nt) if t not in finished][:limit]
    print(f"{year}: {len(finished)} tiles done, {len(todo)} to read with {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        pool.map(_work, [(year, todo[k::workers], k) for k in range(workers)])
    print(f"{year}: {len(todo)} tiles in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "index":
        index()
    else:
        a = sys.argv[2:]
        fetch(int(a[0]), int(a[a.index("--workers") + 1]) if "--workers" in a else 4,
              int(a[a.index("--limit") + 1]) if "--limit" in a else None)
