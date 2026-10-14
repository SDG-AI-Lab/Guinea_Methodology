#!/usr/bin/env python
"""
Build a fast annotation queue for FDI detections over water.

    python 34_build_water_annotation.py --list
    python 34_build_water_annotation.py --date 2024-01-11 --n 150
    python 34_build_water_annotation.py --composite --n 150

Writes manual_qa/annotate_water.html (self-contained, offline) plus the chips
it displays. Open the HTML in a browser, label, then Export CSV. Nothing is
uploaded and progress is kept in the browser's local storage, so the file can
be closed and reopened.

THE QUESTION THIS TOOL ASKS, WHICH IS NOT THE OBVIOUS ONE

The obvious task -- "is this pixel plastic?" -- is not answerable from
Sentinel-2 imagery over water, and asking it would produce a column of
guesses. At 10 m a debris raft is a handful of pixels with no texture, and
the module's own hyperspectral work (Appendix C) established there is no
spectral route to material identity at these scales.

So this tool asks the answerable question instead: WHAT DOES THIS DETECTION
MOST RESEMBLE? Most of the confuser classes are identifiable with real
confidence by eye -- a vessel has a hull and usually a wake, a sediment plume
has a source and a gradient, a cloud has a shadow offset, breaking water sits
on a bathymetric line. Those are the labels worth collecting, and they are
exactly what is needed: section 7.1.13 measured the FDI rule's detections as
38% vessels on a global benchmark, and the open question is whether that
holds on this coast. Confirming the CONFUSERS validates the finding without
anyone having to certify a single pixel as plastic.

"Floating material, cannot identify" and "Uncertain" are therefore first-class
answers, not failures to label. A queue that comes back 40% uncertain is a
useful result: it bounds how much of the detection set is even interpretable.

WHY ONE ROW PER PATCH, NOT PER PIXEL

A detection patch averages a few pixels and is the unit the layer reports.
Pixels within a patch are not independently interpretable -- nobody can say
that pixel 3 of 5 is a hull and pixel 4 is water -- and treating them as
independent would inflate any sample size computed from this queue by the
mean patch size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

MQ = HERE / "manual_qa"
CHIPS = MQ / "water_chips"
from plasticwaste_tool import config as pwcfg              # noqa: E402

# The raster cache the FDI pipeline writes to: <APP_SCRATCH_DIR>/cache/rasters
# (api_connector._cache), from config rather than a hard-coded drive, so the
# script runs on a workstation and in the backend container alike.
CACHE = Path(os.environ.get("PW_RASTER_CACHE")
             or Path(pwcfg.PROCESSING_CONFIG["cache_dir"]) / "rasters")

# Categories, in the order they appear as keys 1-9 (0 for the tenth). They map onto MARIDA's
# classes so the result is directly comparable with section 7.1.13, but they
# are worded for what a human can actually see rather than for what the
# benchmark calls it.
CATEGORIES = [
    ("vessel", "Vessel / boat", "5"),
    ("wake", "Wake or propeller trail", "14"),
    ("sediment", "Sediment or turbid plume", "8,10"),
    ("foam", "Foam, surf or breaking water", "9,12"),
    ("cloud", "Cloud, haze or cloud shadow", "6,13"),
    ("natural", "Floating natural matter (weed, timber)", "2,3,4"),
    # Added after inspecting the first queue: many Conakry detections sit on
    # the boundary of the water mask, in tidal creeks and mangrove channels
    # where a 10 m pixel straddles land and water. That is a mixed-pixel
    # artefact, not floating material, and it is a distinct failure mode from
    # vessels or turbidity -- worth its own key rather than being absorbed
    # into "uncertain", which would hide how common it is.
    ("shoreline", "Shoreline / edge of water mask (mixed pixel)", "-"),
    ("unknown_floating", "Floating material, cannot identify", "1?"),
    ("clear_water", "Nothing visible, looks like plain water", "7,11"),
    ("uncertain", "Cannot tell / image unusable", "-"),
]


def _load_env() -> None:
    env = HERE.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def scene_stretch(bands, water, keys=("B3", "B4", "B8")):
    """
    ONE stretch for the whole scene, computed over water only.

    A per-chip percentile stretch was tried first and is wrong here. Water
    reflectance at Conakry spans roughly 0.034 to 0.069 in the green band --
    a range so narrow that stretching each chip independently turns sensor
    noise into apparent texture, and makes every chip look equally
    structured whether or not anything is present. It also destroys
    comparability: two chips of identical water render differently depending
    on what else happens to be in frame.

    Percentiles are taken over WATER pixels because that is the population
    being judged; including land would put the stretch limits on rooftops and
    render every wave as black.
    """
    out = {}
    for k in keys:
        v = np.asarray(bands[k])
        good = np.isfinite(v) & water
        if not good.any():
            good = np.isfinite(v)
        lo, hi = np.percentile(v[good], [1, 99.5])
        out[k] = (float(lo), float(max(hi, lo + 1e-6)))
    return out


def _gray(band, lims, sl):
    """
    One band to 8-bit grey under a fixed stretch, with NO-DATA made visible.

    A single acquisition covers about 82% of the Conakry AOI, so roughly a
    fifth of every wide chip is outside the swath. Rendering that as black
    makes it indistinguishable from dark water, which would invite an
    annotator to read a swath edge as a feature. It is drawn in a distinct
    blue-grey instead.
    """
    lo, hi = lims
    v = np.asarray(band)[sl].astype("float32")
    nodata = ~np.isfinite(v)
    g = np.clip((np.nan_to_num(v) - lo) / (hi - lo), 0, 1)
    rgb = np.dstack([g, g, g])
    rgb[nodata] = np.array([0.12, 0.16, 0.22])
    return (rgb * 255).astype("uint8")


def _fdi_rgb(fdi, thr, sl):
    """
    The index itself, so the annotator can see what fired.

    Diverging around the scene threshold: blue below, red above. This is the
    only panel that shows the actual evidence -- the grey panels show what a
    human would see, which is a different and equally necessary thing.
    """
    v = np.asarray(fdi)[sl].astype("float32")
    nodata = ~np.isfinite(v)
    scale = max(abs(thr) * 2.0, 1e-4)
    t = np.clip(np.nan_to_num(v) / scale, -1, 1)
    r = np.clip(t, 0, 1)
    b = np.clip(-t, 0, 1)
    g = 1.0 - np.abs(t)
    rgb = np.dstack([0.15 + 0.85 * r, 0.15 + 0.55 * g, 0.15 + 0.85 * b])
    rgb[nodata] = np.array([0.12, 0.16, 0.22])
    return (np.clip(rgb, 0, 1) * 255).astype("uint8")


def build(date: str, use_composite: bool, aoi: str, n_max: int,
          pad: int, seed: int):
    import rasterio
    from PIL import Image
    from scipy import ndimage

    from plasticwaste_tool import api_connector as api

    if use_composite:
        prefix = f"{aoi}_{api.aoi_cache_token(aoi)}_2024-01-01_2024-02-01_composite"
        label = "2024-01 monthly composite"
    else:
        import datetime
        nxt = (datetime.date.fromisoformat(date)
               + datetime.timedelta(days=1)).isoformat()
        prefix = f"{aoi}_{api.aoi_cache_token(aoi)}_{date}_{nxt}_composite"
        label = f"single acquisition {date}"

    path = CACHE / f"{prefix}.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not in the raster cache.\n"
            f"Run 33_single_date_marida.py --dates {date} first (it fetches "
            f"and caches), or pass --composite to use the monthly product.")

    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        names = list(src.descriptions)
        transform = src.transform
        profile = src.profile.copy()
    bands = {nm: arr[i] for i, nm in enumerate(names)}

    wpath = CACHE / f"{prefix}_WATER.tif"
    if wpath.exists():
        with rasterio.open(wpath) as s:
            water = np.nan_to_num(s.read(1), nan=0.0) > 0
    else:
        water = np.ones(bands["B8"].shape, bool)

    # Queue what the layer PUBLISHES: the same detector the pipeline runs
    # (pipeline.run_from_arrays -- the mean+3sigma FDI-and-NDVI rule, then the
    # MARIDA debris classifier when enabled), as patches of at least the
    # minimum mappable area. Annotating anything else would validate a
    # product nobody sees.
    import math

    from plasticwaste_tool import config as pwcfg
    from plasticwaste_tool.fdi_tool import pipeline as P

    bands["WATER"] = water.astype("float32")
    res = P.run_from_arrays(bands, profile, label)
    fdi = res["fdi"]
    thr = float(res["threshold"])
    valid = water & np.isfinite(fdi) & np.isfinite(bands["B8"])
    min_px = max(1, math.ceil(pwcfg.DETECTION_CONFIG["min_patch_area_m2"]
                              / pwcfg.S2_CONFIG["download_scale_m"] ** 2))
    lab, n_all = ndimage.label(res["hotspot_binary"], structure=np.ones((3, 3)))
    big = np.flatnonzero(np.bincount(lab.ravel())[1:] >= min_px) + 1
    flagged = np.isin(lab, big)
    lab, n_patch = ndimage.label(flagged, structure=np.ones((3, 3)))
    detector = ("FDI-and-NDVI rule, then MARIDA classifier"
                if "debris_filter" in res else "FDI-and-NDVI rule")
    print(f"{label}: {int(valid.sum()):,} water px, threshold {thr:.5f}, "
          f"{int(flagged.sum()):,} published px in {n_patch:,} patches ({detector})")
    if n_patch == 0:
        return [], label, thr

    objs = ndimage.find_objects(lab)
    sizes = ndimage.sum(flagged, lab, range(1, n_patch + 1))

    rng = np.random.default_rng(seed)
    order = np.arange(n_patch)
    # Sample across the size range rather than taking the largest: the layer
    # reports small patches too, and a queue of only the biggest would measure
    # the confuser mix of an unrepresentative subset.
    if n_patch > n_max:
        order = rng.choice(order, n_max, replace=False)

    CHIPS.mkdir(parents=True, exist_ok=True)
    for f in CHIPS.glob("*.png"):
        f.unlink()

    lims = scene_stretch(bands, valid)
    H, W = fdi.shape
    records = []

    def _mark(img, y0, y1, x0, x1, k):
        """Outline the flagged pixels in red on an upscaled chip.

        Without it, a wide chip containing one bright feature in a corner
        invites a verdict on whichever feature catches the eye rather than on
        the one that actually fired."""
        px = np.array(img)
        m = (lab[y0:y1, x0:x1] == k + 1)
        if not m.any():
            return img
        mm = np.array(Image.fromarray(m.astype("uint8") * 255)
                      .resize(img.size, Image.NEAREST)) > 127
        edge = mm ^ ndimage.binary_erosion(mm)
        px[edge] = [255, 40, 40]
        return Image.fromarray(px)

    for k in sorted(order):
        sl = objs[k]
        cy = (sl[0].start + sl[0].stop) // 2
        cx = (sl[1].start + sl[1].stop) // 2
        pid = f"w{k:05d}"

        # Three panels, because they answer different questions. Grey green
        # is what a human would see; the wide view places it in its setting
        # (harbour, plume source, bathymetric line); the index panel shows
        # what actually fired, which is often not what the eye is drawn to.
        for tag, half, mode in (("detail", pad, "gray"),
                                ("context", pad * 5, "gray"),
                                ("index", pad, "fdi")):
            y0, y1 = max(cy - half, 0), min(cy + half, H)
            x0, x1 = max(cx - half, 0), min(cx + half, W)
            sel = (slice(y0, y1), slice(x0, x1))
            chip = (_fdi_rgb(fdi, thr, sel) if mode == "fdi"
                    else _gray(bands["B3"], lims["B3"], sel))
            img = Image.fromarray(chip).resize((384, 384), Image.NEAREST)
            if tag != "context":
                img = _mark(img, y0, y1, x0, x1, k)
            img.save(CHIPS / f"{pid}_{tag}.png")

        lon, lat = transform * (cx + 0.5, cy + 0.5)
        records.append({
            "patch_id": pid, "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "n_px": int(sizes[k]),
            "fdi_max": round(float(np.nanmax(np.where(lab == k + 1, fdi, np.nan))), 5),
            "source": f"{label} ({detector})", "threshold": round(thr, 5),
            "detail_m": int(pad * 2 * 10), "context_m": int(pad * 10 * 10),
            "detail": f"water_chips/{pid}_detail.png",
            "context": f"water_chips/{pid}_context.png",
            "index": f"water_chips/{pid}_index.png",
        })

    # Shuffle the PRESENTATION order. Patch ids follow the connected-component
    # scan, which runs top to bottom, so emitting them in id order makes
    # position in the queue a near-perfect proxy for latitude -- in the first
    # round, Spearman -0.99996. Any drift in the annotator's criteria during a
    # session then produces exactly the same pattern as a real geographic
    # split, and the two cannot be separated afterwards (§7.1.14.5). The ids
    # are unchanged, so the spatial analysis is unaffected; only the order in
    # which a human meets them changes.
    rng.shuffle(records)
    print(f"wrote {len(records)} patches, {len(records) * 3} chips to {CHIPS}")
    print("presentation order shuffled, so annotator drift and geography are "
          "not confounded")
    return records, label, thr


HTML = """<!doctype html><meta charset="utf-8">
<title>Water detection annotation</title>
<style>
 body{font:14px system-ui,sans-serif;margin:0;background:#11151a;color:#e8eef5}
 header{padding:10px 16px;background:#0b0e12;border-bottom:1px solid #223;
        display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #pos{font-weight:600}
 .wrap{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
 .pane{background:#161b22;border:1px solid #243;border-radius:8px;padding:10px}
 .pane h3{margin:0 0 8px;font-size:12px;font-weight:600;color:#9fb3c8;
          text-transform:uppercase;letter-spacing:.06em}
 img{display:block;width:384px;height:384px;image-rendering:pixelated;
     border-radius:4px;background:#000}
 .cats{display:flex;flex-direction:column;gap:6px;min-width:330px}
 button.cat{display:flex;justify-content:space-between;align-items:center;
   gap:10px;padding:9px 12px;border-radius:6px;border:1px solid #2c3a4a;
   background:#1b222c;color:#e8eef5;cursor:pointer;font-size:13px;text-align:left}
 button.cat:hover{background:#243040}
 button.cat.sel{background:#1e5f3f;border-color:#2ea06a}
 kbd{background:#0b0e12;border:1px solid #33465c;border-radius:4px;
     padding:1px 7px;font-size:12px;color:#9fb3c8}
 .meta{font-size:12px;color:#9fb3c8;line-height:1.7}
 .primary{background:#2563eb;border:0;color:#fff;padding:8px 14px;
          border-radius:6px;cursor:pointer;font-size:13px}
 .ghost{background:transparent;border:1px solid #33465c;color:#9fb3c8;
        padding:8px 14px;border-radius:6px;cursor:pointer;font-size:13px}
 input[type=text]{background:#0b0e12;border:1px solid #33465c;color:#e8eef5;
   border-radius:6px;padding:8px 10px;width:100%;font-size:13px}
 .bar{height:4px;background:#243040;border-radius:2px;overflow:hidden;flex:1;
      min-width:160px}
 .bar>div{height:100%;background:#2ea06a;width:0}
 .note{max-width:760px;padding:0 16px 20px;color:#9fb3c8;font-size:12.5px;
       line-height:1.65}
</style>
<header>
  <span id="pos">-</span>
  <div class="bar"><div id="prog"></div></div>
  <span id="count" class="meta"></span>
  <button class="ghost" id="prev">&larr; Prev</button>
  <button class="ghost" id="next">Next &rarr;</button>
  <button class="primary" id="exp">Export CSV</button>
</header>
<div class="wrap">
  <div class="pane"><h3>Detection &mdash; green band</h3>
    <img id="detail" alt=""><div class="meta" id="cap1"></div></div>
  <div class="pane"><h3>Context &mdash; 5&times; wider</h3>
    <img id="context" alt=""><div class="meta" id="cap2"></div></div>
  <div class="pane"><h3>FDI &mdash; what fired (red = above threshold)</h3>
    <img id="index" alt=""><div class="meta">blue below, red above</div></div>
  <div class="pane">
    <h3>What does this most resemble?</h3>
    <div class="cats" id="cats"></div>
    <h3 style="margin-top:14px">Note (optional)</h3>
    <input type="text" id="note" placeholder="free text, saved with the label">
    <div class="meta" id="meta" style="margin-top:12px"></div>
  </div>
</div>
<div class="note">
<b>How to use.</b> Press <kbd>1</kbd>&ndash;<kbd>9</kbd> and <kbd>0</kbd> to label, and the
queue advances automatically. <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> move without
labelling; <kbd>U</kbd> jumps to the next unlabelled item. Progress is saved
in this browser, so you can close the page and come back.
<p><b>Answer what you can see, not what you hope is there.</b> You are not
being asked whether anything is plastic &mdash; that is not decidable from
10&nbsp;m imagery, and the analysis does not need it. It needs to know what
share of these detections are vessels, wakes, plumes, foam and cloud. If you
cannot tell, say so: <i>Floating material, cannot identify</i> and
<i>Cannot tell</i> are real answers and the analysis uses them. Guessing
would corrupt the one thing this queue is for.
<p>The <b>context</b> view is there to help with the persistent cases: a
plume usually has a visible source, a shoal recurs along a bathymetric line,
and a vessel near a harbour is more plausible than one in open water.</p>
</div>
<script>
const ITEMS = __ITEMS__;
const CATS  = __CATS__;
const KEY   = "water_annot___STAMP__";
let store = {}, i = 0;
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e){ store = {}; }

const $ = id => document.getElementById(id);

function save(){ try { localStorage.setItem(KEY, JSON.stringify(store)); } catch(e){} }

function build(){
  const box = $("cats"); box.innerHTML = "";
  CATS.forEach((c, k) => {
    const b = document.createElement("button");
    b.className = "cat"; b.dataset.key = c[0];
    b.innerHTML = `<span>${c[1]}</span><kbd>${k===9?0:k+1}</kbd>`;
    b.onclick = () => label(c[0]);
    box.appendChild(b);
  });
}

function render(){
  const it = ITEMS[i], rec = store[it.patch_id] || {};
  $("detail").src = it.detail;
  $("context").src = it.context;
  $("index").src = it.index;
  $("cap1").textContent = `${it.detail_m} m across`;
  $("cap2").textContent = `${it.context_m} m across`;
  $("pos").textContent = `${i+1} / ${ITEMS.length}`;
  $("note").value = rec.note || "";
  document.querySelectorAll("button.cat").forEach(b =>
    b.classList.toggle("sel", b.dataset.key === rec.cat));
  $("meta").innerHTML =
    `patch <b>${it.patch_id}</b><br>${it.n_px} px &middot; max FDI ${it.fdi_max}`
    + `<br>${it.lat.toFixed(5)}, ${it.lon.toFixed(5)}`
    + `<br>${it.source}<br>threshold ${it.threshold}`;
  const done = Object.keys(store).filter(k => store[k].cat).length;
  $("count").textContent = `${done} labelled, ${ITEMS.length-done} left`;
  $("prog").style.width = (100*done/ITEMS.length) + "%";
}

function label(cat){
  const id = ITEMS[i].patch_id;
  store[id] = Object.assign({}, store[id], {cat: cat, note: $("note").value});
  save(); render();
  if (i < ITEMS.length - 1) { i++; render(); }
}

function jumpUnlabelled(){
  for (let k = 1; k <= ITEMS.length; k++){
    const j = (i + k) % ITEMS.length;
    if (!(store[ITEMS[j].patch_id] || {}).cat){ i = j; render(); return; }
  }
}

$("prev").onclick = () => { if (i>0){ i--; render(); } };
$("next").onclick = () => { if (i<ITEMS.length-1){ i++; render(); } };
$("note").addEventListener("change", () => {
  const id = ITEMS[i].patch_id;
  store[id] = Object.assign({}, store[id], {note: $("note").value}); save();
});

window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "0" && CATS.length >= 10) label(CATS[9][0]);
  else if (e.key >= "1" && e.key <= "9" && +e.key <= CATS.length) label(CATS[+e.key-1][0]);
  else if (e.key === "ArrowLeft")  $("prev").click();
  else if (e.key === "ArrowRight") $("next").click();
  else if (e.key.toLowerCase() === "u") jumpUnlabelled();
});

$("exp").onclick = () => {
  const head = ["patch_id","lat","lon","n_px","fdi_max","threshold","source",
                "category","note"];
  const rows = [head.join(",")];
  ITEMS.forEach(it => {
    const r = store[it.patch_id] || {};
    rows.push([it.patch_id, it.lat, it.lon, it.n_px, it.fdi_max, it.threshold,
               `"${it.source}"`, r.cat || "", `"${(r.note||"").replace(/"/g,'""')}"`
              ].join(","));
  });
  const blob = new Blob([rows.join("\\n")], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "water_annotations.csv";
  a.click();
};

build(); render();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", default="conakry")
    ap.add_argument("--date", default="2024-01-11")
    ap.add_argument("--composite", action="store_true",
                    help="use the monthly composite instead of a single date")
    ap.add_argument("--n", type=int, default=150,
                    help="patches to queue (default %(default)s)")
    ap.add_argument("--pad", type=int, default=30,
                    help="half-width of the detail chip, in pixels")
    ap.add_argument("--seed", type=int, default=20240119)
    ap.add_argument("--list", action="store_true",
                    help="show cached rasters that can be annotated")
    args = ap.parse_args()

    _load_env()

    if args.list:
        print(f"cached rasters under {CACHE}:")
        for f in sorted(CACHE.glob(f"{args.aoi}*composite.tif")):
            print(f"  {f.name}")
        return 0

    records, label, thr = build(args.date, args.composite, args.aoi,
                                args.n, args.pad, args.seed)

    # A queue identity. Patch ids (w00012, ...) recur from one build to the
    # next, so without it the page would restore labels a browser saved for an
    # earlier queue, and 35 would score them against the wrong detections. The
    # stamp keys the page's storage, and water_queue.json lets 35 check every
    # exported row against the queue it came from.
    import hashlib
    stamp = hashlib.sha1(json.dumps(
        [label] + sorted((r["patch_id"], r["lat"], r["lon"]) for r in records)).encode()
    ).hexdigest()[:10]
    if not records:
        # Still written, so 35 sees an empty current queue rather than falling
        # back to whatever CSV is on disk from an earlier one.
        print(f"{label}: no published patches -- nothing to annotate. "
              f"An empty queue is recorded so no earlier labels are scored.")
    (MQ / "water_queue.json").write_text(json.dumps({
        "stamp": stamp, "source": label, "threshold": thr,
        "items": [{"patch_id": r["patch_id"], "lat": r["lat"], "lon": r["lon"]}
                  for r in records]}, indent=1), encoding="utf-8")

    out = MQ / "annotate_water.html"
    html = (HTML
            .replace("__ITEMS__", json.dumps(records))
            .replace("__CATS__", json.dumps(CATEGORIES))
            .replace("__STAMP__", stamp))
    out.write_text(html, encoding="utf-8")
    print(f"queue {stamp}: labels saved for any earlier queue are not reused")

    print(f"\nwrote {out}")
    print(f"  open it in a browser, label with keys 1-9 and 0, "
          f"then Export CSV")
    print(f"  save the CSV next to it as manual_qa/water_annotations.csv and "
          f"run 35_score_water_annotation.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
