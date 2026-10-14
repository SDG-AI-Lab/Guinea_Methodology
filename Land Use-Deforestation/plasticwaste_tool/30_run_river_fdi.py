"""
Run the FDI upstream, over the river corridors built by script 29.

WHAT THIS ADDS
--------------
`AOI_CONFIG` holds five hand-drawn coastal boxes. Two of them, Konkoure and
Fatala, are river MOUTHS ("embouchure"), so the delivered layer has until now
covered the coast and the estuaries but nothing upstream. The concept note asks
for accumulation on the coast AND in rivers.

Corridors from script 29 are registered into `AOI_CONFIG` at run time rather
than being pasted into `config.py`, because there are twenty of them, they are
derived from a public dataset by a reproducible rule, and hand-copying
generated geometry into a hand-maintained config is how the two drift apart.

WHY EACH TILE IS ITS OWN AOI, NOT EACH SYSTEM
----------------------------------------------
For the same reason `run_national` splits the coast by region: mean + n*sigma
is an anomaly threshold measured against a background population, and pooling
the Niger headwaters with a mangrove estuary produces one compromise number
that is too high in clear water and too low in turbid water. So each fetchable
unit gets its own threshold, carried on its patches.

The unit had to be smaller than a whole river SYSTEM, though. The first live
run registered systems by the bounding box of their dissolved corridor and one
of them -- a 2,424 km river -- hung for two hours without finishing, because a
winding river's bounding box is mostly empty land: measured across Guinea's 20
systems, corridor-to-bbox bloat ran 10-300x, and the worst single system
implied fetching 61,772 km^2 to analyse a 203 km^2 corridor.

So script 29 tiles each system's reaches into fetchable pieces bounded to
roughly the size of an existing coastal AOI (~2,500 km^2), via
`guinea_river_corridor_tiles.geojson` and `river_aois.json["aois"]`, each
keyed `river_<MAIN_RIV>_t<NN>`. `river_aois.json["systems"]` groups tile keys
back by MAIN_RIV for reporting. A tile's threshold is therefore local to that
tile, not pooled across its whole system -- a known simplification, recorded
in script 29's docstring, and not yet a pooled per-system estimate.

THE GUARD
---------
Every corridor is passed `aoi_kind="river_corridor"`, which tells
`assess_water_population` that a low water FRACTION is expected here and not a
symptom of a misplaced box: a corridor is wide because the HydroRIVERS
centreline is imprecise, not because the analyst drew it wrong. The absolute
`min_water_px` floor is what decides a corridor, and all twenty Guinean systems
clear it by a wide margin on the script-29 estimate. If a real run falls short,
the corridor is reported NOT ASSESSED, which is the honest outcome and is not
the same as "no debris here".

Usage:
    python 30_run_river_fdi.py --list                       # no Earth Engine
    python 30_run_river_fdi.py --start 2024-01-01 --end 2024-02-01 --limit 1
    python 30_run_river_fdi.py --start 2024-01-01 --end 2024-02-01
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

AOIS = HERE / "data" / "rivers" / "river_aois.json"
OUT_DIR = HERE / "manual_qa" / "runs" / "river_fdi"


def load_aois():
    if not AOIS.exists():
        raise SystemExit(
            f"{AOIS} not found. Build the corridors first:\n"
            f"  python 29_prepare_river_corridors.py")
    return json.loads(AOIS.read_text(encoding="utf-8"))


def register(meta):
    """Inject the corridors into AOI_CONFIG for this process only."""
    import plasticwaste_tool.config as pwcfg
    added = []
    for key, a in meta["aois"].items():
        if key in pwcfg.AOI_CONFIG:
            continue
        pwcfg.AOI_CONFIG[key] = {
            "bbox": a["bbox"],
            "label_fr": f"Couloir fluvial {a['main_riv']} "
                        f"({a['length_km']:.0f} km)",
        }
        added.append(key)
    return added


def cmd_list(meta, args=None):
    """Show what would run, and what it is expected to cost. No credentials."""
    aois = meta["aois"]
    systems = meta.get("systems", {})
    print(f"{meta['n_systems']} river systems -> "
          f"{meta.get('n_tiles', len(aois))} fetchable tiles, "
          f"{meta['total_length_km']:,.0f} km, "
          f"{meta['total_corridor_km2']:,.0f} km2 of corridor")
    print(f"total fetch area (sum of tile bboxes): "
          f"{meta.get('total_fetch_bbox_km2', 0):,.0f} km2")
    print(f"source: {meta['source']}")
    print(f"screen: Q >= {meta['min_discharge_cms']:g} m3/s, "
          f"buffer {meta['buffer_m']:g} m, "
          f"max tile {meta.get('max_tile_km2', 0):g} km2\n")

    floor = 1000
    print(f"{'system':>12s} {'km':>7s} {'Q m3/s':>8s} {'tiles':>6s} "
          f"{'est water px':>13s} {'expect':>13s}")
    for mk, s in sorted(systems.items(),
                        key=lambda kv: -kv[1]["max_discharge_cms"]):
        tot_px = sum(aois[k]["est_water_px"] for k in s["tiles"])
        verdict = "assessed" if tot_px >= floor else "NOT assessed"
        print(f"{s['main_riv']:>12d} {s['length_km']:7.0f} "
              f"{s['max_discharge_cms']:8.1f} {s['n_tiles']:6d} "
              f"{tot_px:13,d} {verdict:>13s}")

    print(f"\nest_water_px is a geometric ESTIMATE from channel width and "
          f"length.\nThe real figure comes from the water mask in the "
          f"composite; run --preflight for it (no imagery, no cost).")
    if args is not None and getattr(args, "tiles", False):
        print(f"\n{'aoi key':>22s} {'km':>6s} {'Q m3/s':>8s} "
              f"{'bbox km2':>9s} {'est water px':>13s}")
        for k, a in sorted(aois.items(),
                           key=lambda kv: -kv[1]["max_discharge_cms"]):
            print(f"{k:>22s} {a['length_km']:6.0f} {a['max_discharge_cms']:8.1f} "
                  f"{a['bbox_km2']:9.0f} {a['est_water_px']:13,d}")
    else:
        print(f"Pass --tiles to list all {len(aois)} individual fetchable AOIs.")
    return 0


def cmd_preflight(meta, args):
    """
    Count the water pixels each corridor ACTUALLY has, before running anything.

    Script 29's `est_water_px` is geometry: channel width from discharge, times
    length, divided by pixel size. It says what a river of that size *ought* to
    present. What the pipeline actually sees is the ESA WorldCover
    "permanent water bodies" class, resampled to the download scale, minus a
    20 m dilation around built-up land.

    Those two numbers diverge, and the first real run showed how far. A 13 km
    corridor with an estimated 6,088 water pixels returned FOUR. WorldCover is
    a land-cover product: narrow and seasonal rivers are often not mapped as
    permanent water at all, and in January -- Guinea's dry season -- a 46 m
    channel is about two pixels wide at the 20 m download scale before the
    built-up erosion touches it.

    That is a limitation of the water product, not of the guard, which
    correctly refused to report a threshold estimated from four pixels. But it
    means the useful question is not "which rivers are big enough in principle"
    but "which rivers does the water mask actually see", and that can be
    answered with one cheap reduceRegion per corridor instead of twenty full
    composite downloads.

    Costs no imagery: server-side pixel counts only, nothing is fetched.
    """
    import plasticwaste_tool.config as pwcfg
    from plasticwaste_tool import api_connector as api

    api.initialize_gee()
    ee = api._EE
    scale = args.scale or pwcfg.S2_CONFIG["download_scale_m"]
    floor = pwcfg.DETECTION_CONFIG.get("min_water_px", 1000)

    # Geometry is per TILE, not per system -- see 29's docstring on why a
    # whole system cannot be fetched in one call.
    tiles_path = HERE / "data" / "rivers" / "guinea_river_corridor_tiles.geojson"
    tiles = json.loads(tiles_path.read_text(encoding="utf-8"))
    by_tile = {f["properties"]["tile_key"]: f["geometry"]
              for f in tiles["features"]}

    water_img = api.build_water_mask() if hasattr(api, "build_water_mask") else None
    if water_img is None:
        cfg = pwcfg.DETECTION_CONFIG
        lc = ee.ImageCollection(cfg["water_mask_asset"]).first().select("Map")
        w = lc.eq(cfg["water_mask_class"])
        b = lc.eq(cfg["built_mask_class"])
        if cfg.get("built_buffer_m", 0):
            b = b.focal_max(radius=cfg["built_buffer_m"],
                            kernelType="circle", units="meters")
        water_img = w.And(b.Not()).rename("WATER")

    rows = []
    keys = sorted(meta["aois"],
                  key=lambda k: -meta["aois"][k]["max_discharge_cms"])
    print(f"Counting WorldCover water pixels at {scale} m, "
          f"floor = {floor:,} px\n")
    print(f"{'aoi key':>22s} {'width m':>8s} {'est px':>10s} "
          f"{'actual px':>10s} {'ratio':>7s} {'verdict':>13s}")
    for k in keys:
        a = meta["aois"][k]
        geom = ee.Geometry(by_tile[a["tile_key"]])
        n = water_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geom, scale=scale,
            maxPixels=1e10, bestEffort=True).get("WATER").getInfo()
        n = int(n or 0)
        ok = n >= floor
        rows.append({**{kk: a[kk] for kk in
                        ("main_riv", "length_km", "max_discharge_cms",
                         "est_channel_width_m", "est_water_px")},
                     "aoi": k, "actual_water_px": n,
                     "would_assess": ok})
        ratio = n / max(a["est_water_px"], 1)
        print(f"{k:>22s} {a['est_channel_width_m']:8.0f} "
              f"{a['est_water_px']:10,d} {n:10,d} {ratio:7.2f} "
              f"{('assessable' if ok else 'NOT assessable'):>13s}")

    ok = [r for r in rows if r["would_assess"]]
    out = OUT_DIR / "river_preflight.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scale_m": scale, "min_water_px": floor,
        "water_mask_asset": pwcfg.DETECTION_CONFIG["water_mask_asset"],
        "water_mask_class": pwcfg.DETECTION_CONFIG["water_mask_class"],
        "n_assessable": len(ok), "n_total": len(rows), "rows": rows,
    }, indent=2), encoding="utf-8")

    print(f"\n{len(ok)}/{len(rows)} corridors carry enough mapped water to "
          f"be assessed.")
    if ok:
        print("Run only those:")
        print("  python 30_run_river_fdi.py --start ... --end ... --only "
              + " ".join(r["aoi"] for r in ok))
    print(f"\nWritten to {out}")
    return 0


def _out_path(start, end):
    return OUT_DIR / f"river_fdi_{start}_{end}.json"


def _load_existing(start, end):
    import plasticwaste_tool.config as pwcfg
    p = _out_path(start, end)
    if not p.exists():
        return {"results": [], "failures": []}
    old = json.loads(p.read_text(encoding="utf-8"))
    # A run made by a different detector -- an earlier FDI definition, or
    # without the debris classifier -- must not be resumed: its patches belong
    # to a different product. Start again.
    from plasticwaste_tool.fdi_tool.debris_filter import detector_id
    current = detector_id()
    if old.get("detector") != current:
        print(f"{p.name} was written by detector "
              f"{old.get('detector') or old.get('fdi_formula') or 'unrecorded (B8A variant)'!r}, "
              f"not {current!r}; ignoring it and running every tile again.")
        return {"results": [], "failures": []}
    return old


def _save(start, end, meta, rows, failures):
    import plasticwaste_tool.config as pwcfg
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _out_path(start, end).write_text(json.dumps({
        "start": start, "end": end,
        "fdi_formula": pwcfg.S2_CONFIG.get("fdi_formula"),
        "detector": __import__("plasticwaste_tool.fdi_tool.debris_filter",
                               fromlist=["detector_id"]).detector_id(),
        "n_ok": len(rows), "n_failed": len(failures),
        "min_water_px": pwcfg.DETECTION_CONFIG.get("min_water_px"),
        "min_water_fraction": pwcfg.DETECTION_CONFIG.get("min_water_fraction"),
        "corridor_source": meta["source"],
        "results": rows, "failures": failures,
    }, indent=2), encoding="utf-8")


def cmd_run(meta, args):
    """
    Run the FDI over river tiles.

    WRITES AFTER EVERY TILE, not once at the end. A run over 100+ tiles is a
    long subprocess call and a notebook cell has its own timeout; if that
    kills this process, an end-of-run-only write would lose every tile
    completed up to that point -- which is exactly what happened the first
    time this was run over 16 (then unstiled) AOIs. `--resume` reads what is
    already on disk and skips tile keys already present, so an interrupted run
    costs only the tile it was on, not the ones before it.
    """
    from plasticwaste_tool.fdi_tool import pipeline as P

    added = register(meta)
    keys = list(meta["aois"])
    if args.only:
        keys = [k for k in keys if k in set(args.only)]
    # Smallest first: a cheap tile surfaces an auth or quota problem before an
    # expensive one does.
    keys.sort(key=lambda k: meta["aois"][k]["corridor_km2"])
    if args.limit:
        keys = keys[: args.limit]

    existing = _load_existing(args.start, args.end)
    rows, failures = existing["results"], existing["failures"]
    done = {r["aoi"] for r in rows}
    if args.resume and done:
        before = len(keys)
        keys = [k for k in keys if k not in done]
        print(f"--resume: {before - len(keys)} tile(s) already done this "
              f"run, skipping them")

    print(f"registered {len(added)} tiles; running {len(keys)}\n")

    for i, key in enumerate(keys, 1):
        a = meta["aois"][key]
        print(f"[{i}/{len(keys)}] {key}  "
              f"{a['length_km']:.0f} km, {a['bbox_km2']:.0f} km2 fetch ...",
              flush=True)
        t0 = time.time()
        try:
            res = P.run_from_gee(args.start, args.end, aoi_key=key,
                                 aoi_kind="river_corridor")
        except Exception as exc:  # noqa: BLE001
            print(f"      FAILED: {type(exc).__name__}: {exc}")
            failures.append({"aoi": key,
                             "error": f"{type(exc).__name__}: {exc}"})
            _save(args.start, args.end, meta, rows, failures)
            continue
        # The pipeline stores the vectorised patches under
        # "hotspot_vectors". An earlier version of this script read
        # "hotspots", which does not exist, so `.get` returned None and
        # EVERY tile reported 0 patches regardless of what was found.
        # The run still cost 171 minutes; only the count was wrong.
        gdf = res.get("hotspot_vectors")
        n = 0 if gdf is None else len(gdf)
        assessed = not res.get("low_water_warning")
        rows.append({
            "aoi": key, "main_riv": a["main_riv"],
            "length_km": a["length_km"],
            "start": args.start, "end": args.end,
            "threshold_fdi": res.get("threshold"),
            "threshold_ndvi": res.get("threshold_ndvi"),
            "water_px": res.get("threshold_population_px"),
            "water_fraction": res.get("water_fraction"),
            "low_water_fraction": res.get("low_water_fraction"),
            "low_water_count": res.get("low_water_count"),
            # Why a tile was rejected matters as much as that it was: an empty
            # water mask and an implausible threshold are different failures
            # with different fixes, and both were previously invisible.
            "no_water_mask": res.get("no_water_mask"),
            "threshold_implausible": res.get("threshold_implausible"),
            "assessed": assessed,
            "n_patches": n if assessed else None,
            "seconds": round(time.time() - t0, 1),
        })
        _save(args.start, args.end, meta, rows, failures)  # every tile
        wf = res.get("water_fraction")
        print(f"      {'assessed' if assessed else 'NOT ASSESSED'}: "
              f"{res.get('threshold_population_px') or 0:,} water px, "
              f"{'n/a' if wf is None else f'{wf:.1%}'} of window, "
              f"{n} patches, {rows[-1]['seconds']}s")

    ok = [r for r in rows if r["assessed"]]
    print(f"\n{len(ok)}/{len(rows)} tiles assessed (across every run so "
          f"far for this date range), {len(failures)} failed")
    if ok:
        by_sys = {}
        for r in ok:
            by_sys.setdefault(r["main_riv"], []).append(r)
        print(f"\n{len(by_sys)} river systems with at least one assessed "
              f"tile, {sum(r['n_patches'] for r in ok)} patches total")
        print(f"{'system':>12s} {'tiles ok':>9s} {'patches':>8s} "
              f"{'threshold range':>28s}")
        for mriv, rs in sorted(by_sys.items(), key=lambda kv: -len(kv[1])):
            ths = [r["threshold_fdi"] for r in rs
                  if r["threshold_fdi"] is not None]
            rng = f"{min(ths):.5f} - {max(ths):.5f}" if ths else "n/a"
            print(f"{mriv:>12d} {len(rs):9d} "
                  f"{sum(r['n_patches'] for r in rs):8d} {rng:>28s}")
        print("\nThresholds differ per TILE by design; a patch count is not "
              "comparable\nbetween tiles, or between rivers, unless the "
              "threshold is quoted with it.")
    print(f"\nWritten to {_out_path(args.start, args.end)}")
    return 0 if not failures else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="show the corridors and expected verdicts, no GEE")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD, exclusive upper bound")
    ap.add_argument("--limit", type=int,
                    help="run only the N smallest corridors")
    ap.add_argument("--only", nargs="*", help="specific aoi (tile) keys")
    ap.add_argument("--preflight", action="store_true",
                    help="count real mapped water per tile; no imagery")
    ap.add_argument("--scale", type=float,
                    help="metres; defaults to the download scale")
    ap.add_argument("--resume", action="store_true",
                    help="skip tile keys already in this date range's "
                        "output file")
    ap.add_argument("--tiles", action="store_true",
                    help="with --list, show every individual tile rather "
                        "than the per-system rollup")
    args = ap.parse_args()

    meta = load_aois()
    if args.preflight:
        return cmd_preflight(meta, args)
    if args.list or not (args.start and args.end):
        return cmd_list(meta, args)
    return cmd_run(meta, args)


if __name__ == "__main__":
    raise SystemExit(main())
