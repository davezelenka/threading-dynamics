#!/usr/bin/env python3
"""
build_variability_proxy.py -- range extremes as a proxy for Holocene climate variability.

Subcommands (run in order):
  selftest         checks the bound logic on synthetic data (no downloads)
  fetch-gbif       latitude-stratified random sample of occurrences per taxon (North America)
  clean-occ        drop spatially isolated records (likely planted or mis-georeferenced)
  check-occ        coverage report: extent, box truncation, clustering
  fetch-climate    download WorldClim 2.1 monthly temperature + precipitation into climate/
  extract-climate  sample monthly T/P rasters at occurrences, sites, and reference localities
  breadth          niche breadth per taxon (species-wide and local), range-size control, group comparison
  profile          where a place (default Port Angeles) sits inside each species' climate range
  holocene-maps    maps of WA temperature and Koppen classes at the coldest / warmest Holocene bounds
  analogs          present-day places matching each site's bounded Holocene temperatures
  site-bounds      per-site bounds on Holocene climate excursion from persistent / never-present / lost / gained taxa

Climate rasters: 12 monthly mean-temperature + 12 monthly precipitation GeoTIFFs from ONE
baseline (e.g., ClimateNA 1991-2020 or WorldClim 2.1). Requires numpy, pandas, requests; rasterio
for extract-climate. Outputs go to ../output/.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
# Works whether the script sits next to data/ (flat layout) or inside code/ (zip layout)
ROOT = _HERE if (_HERE / "data").is_dir() else _HERE.parent
DATA, OUT = ROOT / "data", ROOT / "output"
OCC = DATA / "occurrences"
DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
JJA = [5, 6, 7]
VARS = ["MAT", "MTCO", "MTWA", "GDD5", "MAP", "P_JJA", "MI", "MI_JJA"]
LO_Q, HI_Q = 0.05, 0.95

# ----------------------------------------------------------------------------------- climate
def thornthwaite_pet(T, lat):
    T = np.asarray(T, float); Tp = np.where(T > 0, T, 0.0)
    I = np.sum((Tp / 5.0) ** 1.514)
    if I == 0: return np.zeros(12)
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
    doy = np.cumsum(DAYS) - DAYS / 2
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    L = 24 / np.pi * np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(decl), -1, 1))
    return np.where(T > 0, 16 * (L / 12) * (DAYS / 30) * (10 * Tp / I) ** a, 0.0)


def derived(T, P, lat):
    T = np.asarray(T, float); P = np.asarray(P, float); pet = thornthwaite_pet(T, lat)
    noP = np.isnan(P).any()
    return {"MAT": T.mean(), "MTCO": T.min(), "MTWA": T.max(),
            "GDD5": float(np.sum(np.maximum(T - 5, 0) * DAYS)),
            "MAP": np.nan if noP else P.sum(), "P_JJA": np.nan if noP else P[JJA].sum(),
            "MI": np.nan if noP or pet.sum() <= 0 else P.sum() / pet.sum(),
            "MI_JJA": np.nan if noP or pet[JJA].sum() <= 0 else P[JJA].sum() / pet[JJA].sum()}

# ----------------------------------------------------------------------------------- envelopes
def envelope(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    return (x.quantile(LO_Q), x.quantile(HI_Q), len(x)) if len(x) else (np.nan, np.nan, 0)


def site_bounds_one(x0, rows):
    """rows: list of (status, L, U, taxon). Returns bounds on Holocene excursion of x around modern x0.
    Upper bounds (how far it CAN have gone):   from persistent (inside) and never_present (outside) taxa.
    Lower bounds (how far it MUST have gone):  from lost (present -> absent) and gained (absent -> present)."""
    up_cap, dn_cap = [], []            # caps on upward / downward excursion
    up_need, dn_need, any_need = [0.0], [0.0], [0.0]
    flags = []
    for status, L, U, t in rows:
        if np.isnan(L) or np.isnan(U): continue
        if status == "persistent":
            if not (L <= x0 <= U): flags.append(f"{t}: persistent but modern site outside envelope")
            up_cap.append((U - x0, t)); dn_cap.append((x0 - L, t))
        elif status == "never_present":
            if L > x0: up_cap.append((L - x0, t))
            elif U < x0: dn_cap.append((x0 - U, t))
            else: flags.append(f"{t}: never present yet modern site inside envelope (non-climatic absence?)")
        elif status == "lost":          # was present, now absent: past climate was inside, now outside
            if x0 > U: dn_need.append(x0 - U)
            elif x0 < L: up_need.append(L - x0)
            else: flags.append(f"{t}: lost but modern site inside envelope (non-climatic loss?)")
        elif status == "gained":        # was absent, now present: past climate outside, unless dispersal-limited
            any_need.append(min(x0 - L, U - x0))
    up = min(up_cap) if up_cap else (np.nan, "")
    dn = min(dn_cap) if dn_cap else (np.nan, "")
    return {"max_up": up[0], "binding_up": up[1], "max_down": dn[0], "binding_down": dn[1],
            "min_up": max(up_need), "min_down": max(dn_need), "min_either": max(any_need),
            "max_total_range": (up[0] + dn[0]) if up_cap and dn_cap else np.nan,
            "consistent": (max(up_need) <= (up[0] if up_cap else np.inf)) and (max(dn_need) <= (dn[0] if dn_cap else np.inf)),
            "flags": " | ".join(flags)}

# ----------------------------------------------------------------------------------- selftest
def cmd_selftest(a):
    ok = True
    # site at MTWA = 18; persistent taxon envelope 14-21 -> can rise <=3, fall <=4
    # never-present taxon needs >= 20 -> caps rise at 2 (tighter); lost taxon envelope 10-16.5 -> must have fallen >= 1.5
    r = site_bounds_one(18.0, [("persistent", 14, 21, "P1"), ("never_present", 20, 30, "N1"),
                               ("lost", 10, 16.5, "Lo1"), ("gained", 15, 25, "G1")])
    exp = {"max_up": 2.0, "max_down": 4.0, "min_down": 1.5, "min_up": 0.0, "min_either": 3.0, "max_total_range": 6.0}
    for k, v in exp.items():
        good = abs(r[k] - v) < 1e-9; ok &= good
        print(f"{'ok ' if good else 'FAIL'} {k} = {r[k]} (expected {v})")
    print("binding_up =", r["binding_up"], "| consistent =", r["consistent"])
    d = derived([5, 6, 8, 10, 13, 16, 19, 19, 16, 11, 7, 4], [140, 100, 95, 70, 50, 38, 18, 25, 40, 95, 160, 150], 47.6)
    print("derived (synthetic):", {k: round(float(v), 2) for k, v in d.items()})
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1

# ----------------------------------------------------------------------------------- GBIF
def _get_json(session, url, params, tries=4):
    """GET with retries and backoff; returns parsed JSON or raises after the last try."""
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=(10, 60))
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** (k + 1)
            print(f"    request failed ({type(e).__name__}); retry {k + 1}/{tries} in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"GBIF not reachable: {url}")


FIELDS = ["key", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
          "basisOfRecord", "establishmentMeans", "year", "datasetKey"]


def _clean(df, max_unc):
    if df.empty: return df
    unc = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
    df = df[unc.isna() | (unc <= max_unc)]
    df = df[~df["establishmentMeans"].fillna("").str.upper().isin(["INTRODUCED", "MANAGED", "INVASIVE"])]
    df = df[(df["decimalLatitude"].abs() > 1e-6) & (df["decimalLongitude"].abs() > 1e-6)]
    return df.assign(la=df["decimalLatitude"].round(3), lo=df["decimalLongitude"].round(3)).drop_duplicates(["la", "lo"])


def cmd_fetch_gbif(a):
    """Latitude-stratified random sampling: for every 1-degree latitude band that holds records,
    take up to --per-band records from several random offsets, so range edges are always sampled."""
    import requests
    OCC.mkdir(parents=True, exist_ok=True)
    api = "https://api.gbif.org/v1"
    rng = np.random.default_rng(a.seed)
    names = list(pd.read_csv(DATA / "indicator_taxa.csv")["gbif_name"])
    if a.only: names = [n for n in names if a.only.lower() in n.lower()]
    session = requests.Session()
    for i, name in enumerate(names, 1):
        fn = OCC / (name.replace(" ", "_") + ".csv")
        if fn.exists() and not a.refresh:
            print(f"[{i}/{len(names)}] have {fn.name} (use --refresh to redo)", flush=True); continue
        m = _get_json(session, f"{api}/species/match", {"name": name})
        key = m.get("usageKey")
        if key is None: print(f"[{i}/{len(names)}] !! no GBIF match for {name}", flush=True); continue
        print(f"[{i}/{len(names)}] {name} -> GBIF '{m.get('scientificName')}' ({m.get('matchType')}, key {key})", flush=True)
        base = {"taxonKey": key, "hasCoordinate": "true", "hasGeospatialIssue": "false",
                "decimalLongitude": f"{a.lon_min},{a.lon_max}", "occurrenceStatus": "PRESENT"}
        rows, total, nb = [], 0, 0
        for lat in range(int(a.lat_min), int(a.lat_max)):
            p = {**base, "decimalLatitude": f"{lat},{lat + 1}"}
            c = _get_json(session, f"{api}/occurrence/search", {**p, "limit": 0}).get("count", 0)
            if c == 0: continue
            total += c; nb += 1
            reachable = min(c, 100000)                      # GBIF search paging limit
            if reachable <= a.per_band:            # small band: take everything
                pages = [(off, 300) for off in range(0, reachable, 300)]
            else:                                   # large band: several random blocks
                k = max(1, int(np.ceil(a.per_band / a.chunk)))
                starts = rng.choice(reachable - a.chunk, size=k, replace=False)
                pages = [(int(st), a.chunk) for st in sorted(starts)]
            got = 0
            for off, lim in pages:
                r = _get_json(session, f"{api}/occurrence/search", {**p, "limit": lim, "offset": off})
                res = r.get("results", []); got += len(res)
                rows += [{f: o.get(f) for f in FIELDS} for o in res]
            print(f"    lat {lat:>3}-{lat + 1:<3} {c:>7} records, sampled {got:>4}", flush=True)
        df = pd.DataFrame(rows).drop_duplicates("key") if rows else pd.DataFrame(columns=FIELDS)
        n0 = len(df); df = _clean(df, a.max_uncertainty_m)
        df.to_csv(fn, index=False)
        print(f"    {nb} latitude bands, {total} records in box; saved {fn.name}: {n0} sampled -> {len(df)} cleaned", flush=True)


def knn_distance_km(lat, lon, k=5, block=1000):
    """Distance (km) from each point to its k-th nearest other point (haversine, chunked)."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    out = np.empty(len(la))
    for i0 in range(0, len(la), block):
        a1, o1 = la[i0:i0 + block, None], lo[i0:i0 + block, None]
        h = np.sin((la[None, :] - a1) / 2) ** 2 + np.cos(a1) * np.cos(la[None, :]) * np.sin((lo[None, :] - o1) / 2) ** 2
        d = 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        d.sort(axis=1)
        out[i0:i0 + block] = d[:, min(k, d.shape[1] - 1)]   # column 0 is the point itself
    return out


def cmd_clean_occ(a):
    """Drop spatially isolated records (likely planted or mis-georeferenced): a record is kept if its
    k-th nearest neighbour lies within --max-km. Writes data/occurrences_clean/ and a log."""
    src, dst = OCC, DATA / "occurrences_clean"
    dst.mkdir(parents=True, exist_ok=True)
    log = []
    for fn in sorted(src.glob("*.csv")):
        d = pd.read_csv(fn)
        if len(d) <= a.k:
            d.to_csv(dst / fn.name, index=False); continue
        dk = knn_distance_km(d["decimalLatitude"], d["decimalLongitude"], a.k)
        keep = dk <= a.max_km
        dropped = d[~keep]
        d[keep].to_csv(dst / fn.name, index=False)
        for _, r in dropped.iterrows():
            log.append({"taxon": fn.stem.replace("_", " "), "lat": r.decimalLatitude, "lon": r.decimalLongitude,
                        "kth_neighbour_km": round(float(dk[_]), 0) if _ < len(dk) else np.nan})
        print(f"{fn.stem.replace('_', ' '):<30} kept {keep.sum():>5}  dropped {(~keep).sum():>4}"
              + (f"   dropped lat range {dropped.decimalLatitude.min():.1f}-{dropped.decimalLatitude.max():.1f}" if (~keep).any() else ""))
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(log, columns=["taxon", "lat", "lon", "kth_neighbour_km"]).to_csv(OUT / "dropped_isolated_records.csv", index=False)
    print(f"\nwrote {dst} and output/dropped_isolated_records.csv (review: a dropped record may be a real disjunct)")


def _occ_dir():
    c = DATA / "occurrences_clean"
    return c if c.is_dir() and any(c.glob("*.csv")) else OCC


def cmd_check_occ(a):
    """Coverage report: spatial extent, edge truncation, clustering, dataset dominance."""
    rows = []
    print(f"reading {_occ_dir()}")
    for fn in sorted(_occ_dir().glob("*.csv")):
        d = pd.read_csv(fn)
        if d.empty: rows.append({"taxon": fn.stem, "n": 0}); continue
        la, lo = d["decimalLatitude"], d["decimalLongitude"]
        cell = list(zip(np.floor(la), np.floor(lo)))
        top_cell = pd.Series(cell).value_counts(normalize=True).iloc[0]
        top_ds = d["datasetKey"].value_counts(normalize=True).iloc[0] if "datasetKey" in d and d["datasetKey"].notna().any() else np.nan
        edge = []
        if la.max() > a.lat_max - 0.5: edge.append("N")
        if la.min() < a.lat_min + 0.5: edge.append("S")
        if lo.max() > a.lon_max - 0.5: edge.append("E")
        if lo.min() < a.lon_min + 0.5: edge.append("W")
        rows.append({"taxon": fn.stem.replace("_", " "), "n": len(d),
                     "lat_min": round(la.min(), 1), "lat_q05": round(la.quantile(.05), 1),
                     "lat_q95": round(la.quantile(.95), 1), "lat_max": round(la.max(), 1),
                     "lon_min": round(lo.min(), 1), "lon_max": round(lo.max(), 1),
                     "cells_1deg": len(set(cell)), "top_cell_share": round(top_cell, 2),
                     "top_dataset_share": round(top_ds, 2) if top_ds == top_ds else np.nan,
                     "touches_box_edge": "".join(edge),
                     "flag": ("BOX-TRUNCATED " if edge else "") + ("CLUSTERED " if top_cell > 0.25 else "")
                             + ("FEW " if len(d) < 200 else "")})
    r = pd.DataFrame(rows); OUT.mkdir(exist_ok=True); r.to_csv(OUT / "occurrence_coverage.csv", index=False)
    pd.set_option("display.width", 200); print(r.to_string(index=False))
    print("\nBOX-TRUNCATED: range reaches the download box edge (extremes cut off).  CLUSTERED: >25% of records in one 1-degree cell.")

# ----------------------------------------------------------------------------------- climate extraction
WC_MIRRORS = ["https://geodata.ucdavis.edu/climate/worldclim/2_1/base/",
              "https://data.biogeo.ucdavis.edu/data/worldclim/v2.1/base/"]


def _download_resumable(url, dest, tries=3, connect_s=30, read_s=300):
    """Stream url to dest, resuming a partial .part file with an HTTP Range request.
    Retries with backoff. Returns True on a complete download."""
    import requests
    part = dest.with_suffix(".part")
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=(connect_s, read_s), headers=headers) as r:
                if r.status_code == 416:                    # already complete
                    part.replace(dest); return True
                r.raise_for_status()
                if have and r.status_code != 206:           # server ignored Range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                print(f"downloading {url}" + (f"  ({total / 1e6:.0f} MB" + (f", resuming at {have / 1e6:.0f} MB)" if have else ")") if total else ""), flush=True)
                got, last = have, -1
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk); got += len(chunk)
                        pct = int(100 * got / total) if total else -1
                        if total and pct != last and pct % 5 == 0:
                            print(f"    {pct:3d}%  {got / 1e6:,.0f} MB", end="\r", flush=True); last = pct
                print()
                if total and got < total: raise IOError(f"incomplete: {got} of {total} bytes")
            part.replace(dest); return True
        except Exception as e:
            wait = 10 * attempt
            print(f"\n    attempt {attempt}/{tries} failed ({type(e).__name__}); "
                  + (f"retrying in {wait}s (partial download kept)" if attempt < tries else "giving up on this server"), flush=True)
            if attempt < tries: time.sleep(wait)
    return False


def cmd_fetch_climate(a):
    """Download WorldClim 2.1 monthly mean temperature (tavg) and precipitation (prec), 1970-2000,
    and unpack the 24 monthly GeoTIFFs into the climate folder. Re-running skips finished files."""
    import requests, zipfile
    cdir = Path(a.climate_dir)
    if not cdir.is_absolute(): cdir = ROOT / cdir
    cdir.mkdir(parents=True, exist_ok=True)
    for var in ["tavg", "prec"]:
        stem = f"wc2.1_{a.res}_{var}"
        want = [f"{stem}_{m:02d}.tif" for m in range(1, 13)]
        if all((cdir / w).exists() for w in want):
            print(f"{stem}: all 12 monthly files already present in {cdir}"); continue
        zpath = cdir / f"{stem}.zip"
        if not (zpath.exists() and zipfile.is_zipfile(zpath)):
            ok = False
            for base in WC_MIRRORS:
                if _download_resumable(base + f"{stem}.zip", zpath) and zipfile.is_zipfile(zpath):
                    ok = True; break
                if zpath.exists() and not zipfile.is_zipfile(zpath):
                    print("    file is not a valid zip; discarding"); zpath.unlink()
            if not ok:
                sys.exit(f"Could not download {stem}.zip (servers slow or unreachable).\n"
                         f"Rerun the same command later: it resumes where it stopped.\n"
                         f"Or download in a browser:\n  {WC_MIRRORS[0]}{stem}.zip\n"
                         f"save it into {cdir} (leave it zipped) and rerun this command to unpack it.")
        with zipfile.ZipFile(zpath) as z:
            members = {Path(n).name: n for n in z.namelist() if n.lower().endswith(".tif")}
            missing = [w for w in want if w not in members]
            if missing: sys.exit(f"{zpath.name} lacks expected files, e.g. {missing[0]}")
            for w in want:
                if not (cdir / w).exists():
                    with z.open(members[w]) as src, open(cdir / w, "wb") as dst:
                        while True:
                            buf = src.read(1 << 20)
                            if not buf: break
                            dst.write(buf)
            print(f"{stem}: unpacked 12 monthly files into {cdir}")
        if not a.keep_zip: zpath.unlink()
    print(f"\ndone. Next:\n  python {Path(__file__).name} extract-climate")


def find_climate_files(cdir, tpat, ppat):
    """Locate 12 monthly temperature and 12 precipitation GeoTIFFs.
    Looks for the exact names first, then anywhere below cdir (zips often unpack into subfolders),
    then auto-detects any WorldClim 2.1 set (wc2.1_<res>_tavg_MM.tif / wc2.1_<res>_prec_MM.tif)."""
    cdir = Path(cdir).expanduser()
    if not cdir.is_absolute():
        cdir = (Path.cwd() / cdir) if (Path.cwd() / cdir).is_dir() else (ROOT / cdir)
    if not cdir.is_dir():
        alt = ROOT / Path(cdir).name
        if alt.is_dir(): cdir = alt
        else: sys.exit(f"Climate folder not found: {cdir}\n(also tried {alt}). Create it and unzip the WorldClim files into it.")
    tifs = list(cdir.rglob("*.tif"))
    def resolve(pat):
        out = []
        for m in range(1, 13):
            name = pat.format(m=m)
            hit = [p for p in tifs if p.name.lower() == name.lower()]
            out.append(hit[0] if hit else None)
        return out
    T, P = resolve(tpat), resolve(ppat)
    if None not in T and None in P:
        print("NOTE: precipitation maps not found -> TEMPERATURE-ONLY run (MAP, P_JJA, MI, MI_JJA left empty).\n"
              "      When precipitation is available, run fetch-climate and extract-climate again.")
        return T, None
    if None in T or None in P:                       # try auto-detect
        import re
        sets = {}
        for p in tifs:
            mt = re.match(r"(wc2\.1_[^_]+)_(tavg|prec)_(\d{2})\.tif$", p.name, re.I)
            if mt: sets.setdefault(mt.group(1), {}).setdefault(mt.group(2).lower(), {})[int(mt.group(3))] = p
        full = [k for k, v in sets.items() if len(v.get("tavg", {})) == 12 and len(v.get("prec", {})) == 12]
        if full:
            k = sorted(full)[0]
            print(f"using auto-detected WorldClim set '{k}' in {cdir}")
            T = [sets[k]["tavg"][m] for m in range(1, 13)]; P = [sets[k]["prec"][m] for m in range(1, 13)]
        else:
            msg = [f"Could not find 12 temperature + 12 precipitation GeoTIFFs under {cdir}",
                   f"looked for e.g. {tpat.format(m=1)} and {ppat.format(m=1)}",
                   f"found {len(tifs)} .tif files" + (":" if tifs else " (folder is empty or files are still zipped)")]
            msg += [f"   {p.relative_to(cdir)}" for p in sorted(tifs)[:15]]
            if len(tifs) > 15: msg.append("   ...")
            for k, v in sets.items():
                msg.append(f"partial set {k}: {len(v.get('tavg', {}))} tavg, {len(v.get('prec', {}))} prec files")
            sys.exit("\n".join(msg))
    return T, P


def sample_climate(df, latc, lonc, files):
    import rasterio
    Tf, Pf = files
    xy = list(zip(df[lonc].values, df[latc].values))
    def stack(paths):
        cols = []
        for path in paths:
            with rasterio.open(path) as r:
                v = np.array([s[0] for s in r.sample(xy)], float)
                if r.nodata is not None: v[v == r.nodata] = np.nan
                v[v < -1e30] = np.nan
                cols.append(v)
        return np.column_stack(cols)
    T = stack(Tf)
    P = stack(Pf) if Pf is not None else np.full_like(T, np.nan)
    bad = np.isnan(T).any(1) | (np.isnan(P).any(1) if Pf is not None else False)
    recs = [derived(T[i], P[i], df[latc].values[i]) if not bad[i] else {} for i in range(len(df))]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)[~bad]


def cmd_extract(a):
    files = find_climate_files(a.climate_dir, a.tpat, a.ppat)
    print(f"temperature: {files[0][0].name} ... {files[0][-1].name}")
    print(f"precipitation: {files[1][0].name} ... {files[1][-1].name}" if files[1] else "precipitation: none (temperature-only)")
    (OUT / "occ_climate").mkdir(parents=True, exist_ok=True)
    print(f"occurrences from {_occ_dir()}")
    for fn in sorted(_occ_dir().glob("*.csv")):
        d = pd.read_csv(fn)
        if not len(d): print(f"climate: {fn.stem}: no records, skipped", flush=True); continue
        out = sample_climate(d, "decimalLatitude", "decimalLongitude", files)
        out.to_csv(OUT / "occ_climate" / fn.name, index=False)
        print(f"climate: {fn.stem:<30} {len(out):>5} of {len(d):>5} records have climate values", flush=True)
    for name in ["paleo_sites.csv", "modern_reference_localities.csv"]:
        if not (DATA / name).exists():
            print(f"skip {name} (not found; optional)"); continue
        d = pd.read_csv(DATA / name); d = d[pd.to_numeric(d["lat"], errors="coerce").notna()].astype({"lat": float, "lon": float})
        sample_climate(d, "lat", "lon", files).to_csv(OUT / name.replace(".csv", "_climate.csv"), index=False)
        print(f"climate: {name}")

# ----------------------------------------------------------------------------------- breadth
def _require_occ_climate():
    """Stop with a clear message if extract-climate has not produced usable files."""
    cdir = OUT / "occ_climate"
    files = sorted(cdir.glob("*.csv")) if cdir.is_dir() else []
    if not files:
        sys.exit(f"No climate-annotated occurrences in {cdir}.\n"
                 f"Run these first:\n  python {Path(__file__).name} fetch-climate\n  python {Path(__file__).name} extract-climate")
    return files


def cmd_breadth(a):
    taxa = pd.read_csv(DATA / "indicator_taxa.csv").set_index("gbif_name")
    files = _require_occ_climate()
    print(f"{len(files)} species files in {OUT / 'occ_climate'}")
    rows = []
    for fn in files:
        name = fn.stem.replace("_", " "); d = pd.read_csv(fn)
        missing = [v for v in VARS if v not in d.columns]
        if missing: print(f"skip {name}: columns missing {missing} (rerun extract-climate)"); continue
        d = d.dropna(subset=["MAT"])
        if len(d) < a.min_n: print(f"skip {name}: only {len(d)} records with climate (min {a.min_n})"); continue
        cells = len(set(zip(np.floor(d.decimalLatitude / 0.5), np.floor(d.decimalLongitude / 0.5))))
        band = np.floor(d.decimalLatitude).astype(int)
        for v in VARS:
            L, U, n = envelope(d[v])
            if n == 0: continue
            # local breadth: median of within-1-degree-band breadths (removes much of the spatial gradient)
            lb = [envelope(g[v])[1] - envelope(g[v])[0] for _, g in d.groupby(band) if len(g) >= 20]
            rows.append({"taxon": name, "group": taxa.loc[name, "group"] if name in taxa.index else "",
                         "variable": v, "n": n, "L05": L, "U95": U, "breadth": U - L,
                         "local_breadth": np.median(lb) if lb else np.nan, "range_cells_0p5deg": cells})
    b = pd.DataFrame(rows)
    if b.empty:
        sys.exit("No species had enough records with climate values. Check output/occ_climate files "
                 "(open one: MAT/MAP columns should hold numbers) and rerun extract-climate.")
    # range-size control (Slatyer et al. 2013): residual of breadth on log range size, per variable
    b["breadth_resid"] = np.nan
    for v, g in b.groupby("variable"):
        ok = g["breadth"].notna()
        if ok.sum() >= 5:
            X = np.log(g.loc[ok, "range_cells_0p5deg"]); c = np.polyfit(X, g.loc[ok, "breadth"], 1)
            b.loc[g.index[ok], "breadth_resid"] = g.loc[ok, "breadth"] - np.polyval(c, X)
    OUT.mkdir(exist_ok=True); b.to_csv(OUT / "niche_breadth.csv", index=False)
    use = set(taxa.index[taxa.get("use_in_breadth", pd.Series(True, index=taxa.index)).astype(str).str.lower() != "false"])
    excluded = sorted(set(b.taxon) - use)
    if excluded: print("excluded from group comparison (use_in_breadth = false):", ", ".join(excluded))
    bg = b[b.taxon.isin(use) & b.group.isin(["west_lowland", "east_lowland", "alpine_reference"])]
    summ = bg.groupby(["variable", "group"])[["breadth", "local_breadth", "breadth_resid"]].median().round(2)
    summ.to_csv(OUT / "niche_breadth_by_group.csv"); print(summ.to_string())
    # permutation test of median differences between groups (labels shuffled)
    rng = np.random.default_rng(1); rows = []
    pairs = [("east_lowland", "west_lowland"), ("west_lowland", "alpine_reference"), ("east_lowland", "alpine_reference")]
    for v in [v for v in VARS if v in set(bg.variable)]:
        for col in ["breadth", "local_breadth", "breadth_resid"]:
            dv = bg[bg.variable == v]
            for g1, g2 in pairs:
                x = dv[dv.group == g1][col].dropna().values; y = dv[dv.group == g2][col].dropna().values
                if len(x) < 3 or len(y) < 3: continue
                obs = np.median(x) - np.median(y); z = np.r_[x, y]; c = 0
                for _ in range(a.n_perm):
                    rng.shuffle(z); c += abs(np.median(z[:len(x)]) - np.median(z[len(x):])) >= abs(obs)
                rows.append({"variable": v, "measure": col, "contrast": f"{g1} - {g2}", "n1": len(x), "n2": len(y),
                             "median_diff": round(obs, 3), "p_perm": round((c + 1) / (a.n_perm + 1), 4)})
    pt = pd.DataFrame(rows); pt.to_csv(OUT / "niche_breadth_permutation.csv", index=False)
    if len(pt):
        print("\npermutation tests (two-sided, median difference):")
        print(pt.to_string(index=False))
        print(f"{len(pt)} tests; Bonferroni threshold at alpha 0.05 = {0.05 / len(pt):.4f} (tests are correlated, so this is strict)")

# ----------------------------------------------------------------------------------- site bounds
def _hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, np.asarray(la2, float), np.asarray(lo2, float)])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


TEMP_VARS = {"MAT", "MTCO", "MTWA", "GDD5"}


def limiting_vars(spec):
    """Parse indicator_taxa.variables_bounded into the set of output variables a taxon is thought to be
    limited by. 'MI' also covers MI_JJA; any temperature variable also covers MAT."""
    import re
    if not isinstance(spec, str): return set()
    toks = set(re.findall(r"[A-Z][A-Z0-9_]+", spec.split("(not")[0]))
    out = {t for t in toks if t in VARS}
    if "MI" in out: out.add("MI_JJA")
    if out & TEMP_VARS: out.add("MAT")
    return out


def _available_climate(sla, slo, radius_km, files, n=4000, seed=0):
    """Climate of random land points within radius_km of a site: what climates EXIST nearby.
    If a taxon's envelope edge sits at the edge of available climate, the edge is set by geography
    (e.g. the ocean), not by tolerance, and a bound resting on it is not informative."""
    rng = np.random.default_rng(seed)
    r = radius_km * np.sqrt(rng.uniform(0, 1, n)); th = rng.uniform(0, 2 * np.pi, n)
    dlat = (r * np.cos(th)) / 111.0
    dlon = (r * np.sin(th)) / (111.0 * np.cos(np.radians(sla)))
    pts = pd.DataFrame({"lat": sla + dlat, "lon": slo + dlon})
    return sample_climate(pts, "lat", "lon", files)


def cmd_site_bounds(a):
    """Bounds from species-wide AND regional envelopes (records within --regional-km of the site).
    Each bound uses the quantile that makes it conservative:
      persistent / lost / gained -> wide envelope (q_wide), never_present -> narrow envelope (q_narrow).
    Regional bounds are also checked against the climate that actually exists within the radius."""
    _require_occ_climate()
    if not (OUT / "paleo_sites_climate.csv").exists():
        sys.exit("output/paleo_sites_climate.csv missing: run extract-climate first.")
    st = pd.read_csv(DATA / "taxon_site_status.csv")
    if not a.include_low_confidence:
        dropped = st[st["dispersal_limited_risk"].str.upper().str.startswith("HIGH")]
        for r in dropped.itertuples(): print(f"excluded (HIGH risk): {r.site_id} {r.gbif_name} {r.status}")
        st = st[~st["dispersal_limited_risk"].str.upper().str.startswith("HIGH")]
    sites = pd.read_csv(OUT / "paleo_sites_climate.csv").set_index("site_id")
    occ = {fn.stem.replace("_", " "): pd.read_csv(fn) for fn in (OUT / "occ_climate").glob("*.csv")}
    taxa = pd.read_csv(DATA / "indicator_taxa.csv")
    lim = {r.gbif_name: limiting_vars(r.variables_bounded) for r in taxa.itertuples()}
    qw, qn = a.q_wide, a.q_narrow
    files = None
    if not a.no_geo_check:
        try: files = find_climate_files(a.climate_dir, a.tpat, a.ppat)
        except SystemExit: print("climate maps not found: geographic-truncation check skipped")

    def env_for(d, status):
        q = qn if status == "never_present" else qw
        out = {}
        for v in VARS:
            x = d[v].replace([np.inf, -np.inf], np.nan).dropna()
            out[v] = (x.quantile(q), x.quantile(1 - q)) if len(x) else (np.nan, np.nan)
        return out

    out = []
    for sid, g in st.groupby("site_id"):
        if sid not in sites.index: print(f"no climate for site {sid}"); continue
        sla, slo = sites.loc[sid, "lat"], sites.loc[sid, "lon"]
        avail = None
        if files is not None:
            av = _available_climate(sla, slo, a.regional_km, files)
            avail = {v: (av[v].quantile(0.01), av[v].quantile(0.99)) for v in VARS if av[v].notna().any()}
        for scope in ["species", "regional"]:
            env = {}
            for r in g.itertuples():
                d = occ.get(r.gbif_name)
                if d is None: continue
                if scope == "regional":
                    d = d[_hav_km(sla, slo, d["decimalLatitude"], d["decimalLongitude"]) <= a.regional_km]
                    if len(d) < a.min_regional_n: continue
                env[(r.gbif_name, r.status)] = env_for(d, r.status)
            for v in VARS:
                if pd.isna(sites.loc[sid, v]): continue
                rows = [(r.status, *env.get((r.gbif_name, r.status), {}).get(v, (np.nan, np.nan)), r.gbif_name)
                        for r in g.itertuples()]
                res_v = site_bounds_one(sites.loc[sid, v], rows)
                res_v["up_set_by_limiting_taxon"] = (v in lim.get(res_v["binding_up"], set())) if res_v["binding_up"] else np.nan
                res_v["down_set_by_limiting_taxon"] = (v in lim.get(res_v["binding_down"], set())) if res_v["binding_down"] else np.nan
                res_v["up_geo_truncated"] = res_v["down_geo_truncated"] = np.nan
                if scope == "regional" and avail and v in avail:
                    lo_av, hi_av = avail[v]; tol = 0.02 * (hi_av - lo_av)
                    stat = {r.gbif_name: r.status for r in g.itertuples()}
                    tu, td = res_v["binding_up"], res_v["binding_down"]
                    if tu and stat.get(tu) == "persistent":
                        res_v["up_geo_truncated"] = bool(env[(tu, "persistent")][v][1] >= hi_av - tol)
                    if td and stat.get(td) == "persistent":
                        res_v["down_geo_truncated"] = bool(env[(td, "persistent")][v][0] <= lo_av + tol)
                out.append({"site_id": sid, "scope": scope, "variable": v, "modern": sites.loc[sid, v], **res_v})
    res = pd.DataFrame(out)
    if res.empty: sys.exit("No bounds computed: check that site taxa have occurrence files with climate values.")
    res.to_csv(OUT / "site_variability_bounds.csv", index=False)

    def verdict(r, side):
        cap = r[f"max_{side}"]
        if pd.isna(cap): return ""
        def yes(x): return (not pd.isna(x)) and bool(x)
        if yes(r[f"{side}_geo_truncated"]): return "GEO"             # edge of available land climate
        return "ok" if yes(r[f"{side}_set_by_limiting_taxon"]) else "prov"
    res["dn"] = res.apply(lambda r: verdict(r, "down"), axis=1)
    res["up"] = res.apply(lambda r: verdict(r, "up"), axis=1)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    print(res[["site_id", "scope", "variable", "modern", "max_down", "dn", "binding_down",
               "max_up", "up", "binding_up", "min_down", "min_up"]].round(2).to_string(index=False))
    print(f"\nquantiles: persistent/lost/gained q{qw:.2f}-q{1 - qw:.2f} (conservative wide); never_present q{qn:.2f}-q{1 - qn:.2f}")
    print("ok   = set by a taxon thought to be limited by this variable")
    print("prov = provisional: set by a taxon limited by something else (method check C8)")
    print("GEO  = not informative: the taxon's edge is the edge of climate that exists nearby (method check C4)")
    fl = res[res["flags"].fillna("").astype(bool)]
    if len(fl): print("\nFLAGS:"); print(fl[["site_id", "scope", "variable", "flags"]].to_string(index=False))


# ----------------------------------------------------------------------------------- Holocene maps
# Approximate Washington State outline (lon, lat), for masking and drawing only.
WA_OUTLINE = [(-124.75, 48.40), (-123.25, 48.22), (-123.12, 48.42), (-123.25, 48.70), (-123.02, 48.83),
              (-123.32, 49.00), (-117.03, 49.00), (-117.04, 46.42), (-116.92, 46.00), (-118.98, 46.00),
              (-119.60, 45.93), (-120.20, 45.72), (-120.90, 45.65), (-121.20, 45.60), (-121.90, 45.65),
              (-122.25, 45.55), (-122.76, 45.65), (-122.80, 45.95), (-123.20, 46.18), (-123.90, 46.25),
              (-124.20, 46.40), (-124.20, 47.00), (-124.50, 47.60), (-124.75, 48.10), (-124.75, 48.40)]
# Approximate Cascade crest (lon, lat): sites and bounds come from west of it.
CASCADE_CREST = [(-121.05, 49.00), (-121.15, 48.50), (-121.30, 47.80), (-121.40, 47.20), (-121.50, 46.70),
                 (-121.60, 46.20), (-121.85, 45.70)]
WEST_SITES = {"BGL", "LWA", "KIL", "PEN", "HOH", "WEN", "CRO", "DAV"}


def koppen(T, P, cd_threshold=0.0):
    """Full Koppen-Geiger class (Peel et al. 2007 rules). E, then B, then A/C/D."""
    T = np.asarray(T, float); P = np.asarray(P, float)
    MAT, MAP, Thot, Tcold = T.mean(), P.sum(), T.max(), T.min()
    if Thot < 10: return "ET" if Thot > 0 else "EF"
    SUM, WIN = [3, 4, 5, 6, 7, 8], [9, 10, 11, 0, 1, 2]
    Psum, Pwin = P[SUM].sum(), P[WIN].sum()
    pth = 2 * MAT if Pwin >= 0.7 * MAP else (2 * MAT + 28 if Psum >= 0.7 * MAP else 2 * MAT + 14)
    if MAP < 10 * pth: return "B" + ("W" if MAP < 5 * pth else "S") + ("h" if MAT >= 18 else "k")
    if Tcold >= 18: return "A"
    main = "C" if Tcold > cd_threshold else "D"
    Psdry, Pswet, Pwdry, Pwwet = P[SUM].min(), P[SUM].max(), P[WIN].min(), P[WIN].max()
    s_ = (Psdry < 40) and (Psdry < Pwwet / 3); w_ = Pwdry < Pswet / 10
    second = ("s" if Pwin > Psum else "w") if (s_ and w_) else ("s" if s_ else ("w" if w_ else "f"))
    return main + second + _third(T, main)


def _third(T, main):
    Thot, Tcold = T.max(), T.min()
    if Thot >= 22: return "a"
    if (T >= 10).sum() >= 4: return "b"
    if main == "D" and Tcold <= -38: return "d"
    return "c"


def thermal_koppen(T, cd_threshold=0.0):
    """Temperature-only Koppen: E vs C vs D and the a/b/c summer letter. The precipitation letter
    (f/s/w) and the arid B class need precipitation and are not determined."""
    T = np.asarray(T, float)
    if T.max() < 10: return "ET" if T.max() > 0 else "EF"
    main = "C" if T.min() > cd_threshold else "D"
    return main + _third(T, main)


def derive_scenarios(bounds_csv):
    """Uniform-shift scenarios from site bounds (western sites, warmest-month mean).
    Cold: smallest downward cap set by a temperature-limited taxon ('ok').
    Warm: smallest upward cap not flagged GEO (provisional unless set by a temperature-limited taxon)."""
    b = pd.read_csv(bounds_csv)
    b = b[(b.variable == "MTWA") & b.site_id.isin(WEST_SITES)]
    def is_true(x): return (not pd.isna(x)) and str(x).lower() in ("true", "1", "1.0")
    cold = b[b.max_down.notna() & b.down_set_by_limiting_taxon.map(is_true) & ~b.down_geo_truncated.map(is_true)]
    warm = b[b.max_up.notna() & ~b.up_geo_truncated.map(is_true)]
    if cold.empty or warm.empty: return None
    c = cold.loc[cold.max_down.idxmin()]; w = warm.loc[warm.max_up.idxmin()]
    return {"cold": -float(c.max_down), "cold_src": f"{c.site_id} {c.scope}: {c.binding_down} (temperature-limited)",
            "warm": float(w.max_up), "warm_src": f"{w.site_id} {w.scope}: {w.binding_up} "
            + ("(temperature-limited)" if is_true(w.up_set_by_limiting_taxon) else "(PROVISIONAL: not temperature-limited)")}


def _read_window(paths, bbox):
    import rasterio
    from rasterio.windows import from_bounds
    lon0, lat0, lon1, lat1 = bbox
    arrs = []
    with rasterio.open(paths[0]) as r0:
        win = from_bounds(lon0, lat0, lon1, lat1, r0.transform).round_offsets().round_lengths()
        tr = r0.window_transform(win)
    for p in paths:
        with rasterio.open(p) as r:
            a = r.read(1, window=win).astype("float64")
            if r.nodata is not None: a[a == r.nodata] = np.nan
            a[a < -1e30] = np.nan
            arrs.append(a)
    stack = np.stack(arrs)                                   # (12, rows, cols)
    rows, cols = stack.shape[1:]
    lons = tr.c + tr.a * (np.arange(cols) + 0.5)
    lats = tr.f + tr.e * (np.arange(rows) + 0.5)
    return stack, lons, lats


def cmd_holocene_maps(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MPath
    from matplotlib.colors import ListedColormap, BoundaryNorm
    files = find_climate_files(a.climate_dir, a.tpat, a.ppat)
    have_p = files[1] is not None
    # ---- scenario shifts
    sc = derive_scenarios(OUT / "site_variability_bounds.csv") if (OUT / "site_variability_bounds.csv").exists() else None
    d_cold = a.d_cold if a.d_cold is not None else (sc["cold"] if sc else None)
    d_warm = a.d_warm if a.d_warm is not None else (sc["warm"] if sc else None)
    if d_cold is None or d_warm is None:
        sys.exit("No scenarios: run site-bounds first, or pass --d-cold and --d-warm (deg C).")
    print(f"cold scenario: {d_cold:+.2f} C" + (f"   <- {sc['cold_src']}" if sc and a.d_cold is None else "   (user value)"))
    print(f"warm scenario: {d_warm:+.2f} C" + (f"   <- {sc['warm_src']}" if sc and a.d_warm is None else "   (user value)"))
    # monthly shape: Jan anomaly = winter_scale * summer anomaly, cosine between (1 = uniform shift)
    shape = (1 + a.winter_scale) / 2 + (1 - a.winter_scale) / 2 * np.cos((np.arange(12) - 6) / 12 * 2 * np.pi)
    # ---- read climate for Washington
    bbox = (-125.0, 45.4, -116.8, 49.15)
    T, lons, lats = _read_window(files[0], bbox)
    P = _read_window(files[1], bbox)[0] if have_p else None
    LON, LAT = np.meshgrid(lons, lats)
    inside = MPath(WA_OUTLINE).contains_points(np.c_[LON.ravel(), LAT.ravel()]).reshape(LON.shape)
    valid = inside & ~np.isnan(T).any(0)
    crest = np.interp(LAT, [p[1] for p in CASCADE_CREST][::-1], [p[0] for p in CASCADE_CREST][::-1])
    east = LON > crest
    scen = {"Coldest Holocene (bound)": d_cold, "Present (1970-2000)": 0.0, "Warmest Holocene (bound)": d_warm}
    OUT.mkdir(exist_ok=True)
    # ---- classify each scenario
    classes, temps, tables = {}, {}, []
    for name, dT in scen.items():
        Ts = T + dT * shape[:, None, None]
        K = np.full(T.shape[1:], "", dtype=object)
        for i, j in zip(*np.where(valid)):
            K[i, j] = koppen(Ts[:, i, j], P[:, i, j], a.cd) if have_p else thermal_koppen(Ts[:, i, j], a.cd)
        classes[name] = K
        temps[name] = {"MTWA": np.where(valid, Ts.max(0), np.nan), "MTCO": np.where(valid, Ts.min(0), np.nan),
                       "MAT": np.where(valid, Ts.mean(0), np.nan)}
        w = np.cos(np.radians(LAT))                     # cell area weight
        for side, m in [("statewide", valid), ("west of crest", valid & ~east), ("east of crest", valid & east)]:
            tot = w[m].sum()
            for k in sorted(set(K[m])):
                tables.append({"scenario": name, "shift_C": dT, "region": side, "class": k,
                               "area_pct": round(100 * w[m & (K == k)].sum() / tot, 1)})
    tab = pd.DataFrame(tables); tab.to_csv(OUT / "holocene_koppen_area.csv", index=False)
    # cells that change class relative to present
    pres = classes["Present (1970-2000)"]
    chg = []
    for name in scen:
        if name.startswith("Present"): continue
        diff = valid & (classes[name] != pres)
        for side, m in [("west of crest", ~east), ("east of crest", east)]:
            chg.append({"scenario": name, "region": side,
                        "pct_area_changing_class": round(100 * np.cos(np.radians(LAT))[diff & m].sum()
                                                         / np.cos(np.radians(LAT))[valid & m].sum(), 1)})
    chg = pd.DataFrame(chg); chg.to_csv(OUT / "holocene_koppen_change.csv", index=False)
    # ---- figures
    ol = np.array(WA_OUTLINE); cr = np.array(CASCADE_CREST)
    def deco(ax, title):
        ax.plot(ol[:, 0], ol[:, 1], color="k", lw=0.7); ax.plot(cr[:, 0], cr[:, 1], "k--", lw=0.8)
        ax.set_title(title, fontsize=10); ax.set_xlim(-124.9, -116.8); ax.set_ylim(45.5, 49.1)
        ax.set_aspect(1 / np.cos(np.radians(47.3))); ax.tick_params(labelsize=7)
    for var, lab, cmap in [("MTWA", "Warmest-month mean (\u00b0C)", "RdYlBu_r"),
                           ("MTCO", "Coldest-month mean (\u00b0C)", "RdYlBu_r"),
                           ("MAT", "Mean annual temperature (\u00b0C)", "RdYlBu_r")]:
        allv = np.concatenate([temps[n][var][valid] for n in scen])
        vmin, vmax = np.nanpercentile(allv, 1), np.nanpercentile(allv, 99)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        for ax, (name, dT) in zip(axs, scen.items()):
            im = ax.pcolormesh(LON, LAT, temps[name][var], cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            ax.contourf(LON, LAT, (east & valid).astype(float), levels=[0.5, 1.5], hatches=["///"], colors="none")
            deco(ax, f"{name}\nshift {dT:+.1f} \u00b0C")
        fig.colorbar(im, ax=axs, shrink=0.85, label=lab)
        fig.suptitle(f"Washington {lab}: Holocene bounds vs present (WorldClim 2.1 baseline). "
                     "Hatched: east of Cascade crest, extrapolated (no informative east-side bounds).", fontsize=9)
        fig.savefig(OUT / f"holocene_map_{var}.png", dpi=170); plt.close(fig)
    allk = sorted(set().union(*[set(classes[n][valid]) for n in scen]))
    palette = {"Ca": "#d7301f", "Cb": "#fc8d59", "Cc": "#fdcc8a", "Da": "#6a51a3", "Db": "#9e9ac8", "Dc": "#cbc9e2",
               "Dd": "#54278f", "ET": "#bdbdbd", "EF": "#ffffff", "Csa": "#e31a1c", "Csb": "#fd8d3c", "Csc": "#fed976",
               "Cfa": "#1a9850", "Cfb": "#66bd63", "Cfc": "#a6d96a", "Dsa": "#3f007d", "Dsb": "#6a51a3", "Dsc": "#9e9ac8",
               "Dfa": "#08519c", "Dfb": "#3182bd", "Dfc": "#9ecae1", "BSk": "#e6ab02", "BWk": "#ffff99", "BSh": "#d95f02"}
    cols = [palette.get(k, "#888888") for k in allk]
    cmapk = ListedColormap(cols); norm = BoundaryNorm(np.arange(len(allk) + 1) - 0.5, len(allk))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (name, dT) in zip(axs, scen.items()):
        idx = np.full(T.shape[1:], np.nan)
        for n_, k in enumerate(allk): idx[classes[name] == k] = n_
        idx[~valid] = np.nan
        im = ax.pcolormesh(LON, LAT, idx, cmap=cmapk, norm=norm, shading="auto")
        ax.contourf(LON, LAT, (east & valid).astype(float), levels=[0.5, 1.5], hatches=["///"], colors="none")
        deco(ax, f"{name}\nshift {dT:+.1f} \u00b0C")
    cb = fig.colorbar(im, ax=axs, shrink=0.85, ticks=range(len(allk))); cb.ax.set_yticklabels(allk)
    kind = "Koppen-Geiger (precipitation held at present)" if have_p else \
        "Thermal Koppen (temperature letters only; precipitation letter and arid B class not determined)"
    fig.suptitle(f"Washington {kind}, C|D at {a.cd:g} \u00b0C. Hatched: east of crest, extrapolated.", fontsize=9)
    fig.savefig(OUT / "holocene_map_koppen.png", dpi=170); plt.close(fig)
    # ---- console summary
    pd.set_option("display.width", 200)
    print("\nKoppen class area (% of region):")
    piv = tab.pivot_table(index=["region", "class"], columns="scenario", values="area_pct", fill_value=0)
    print(piv[[n for n in scen]].to_string())
    print("\n% of area that changes class relative to present:"); print(chg.to_string(index=False))
    print(f"\nwrote {OUT}: holocene_map_MTWA.png, holocene_map_MTCO.png, holocene_map_MAT.png, holocene_map_koppen.png,")
    print("      holocene_koppen_area.csv, holocene_koppen_change.csv")
    if not have_p: print("NOTE: thermal Koppen only. Add precipitation (fetch-climate) for full classes; precipitation is then held at present.")


def cmd_analogs(a):
    """For each site, find where today's climate (MAT, MTCO, MTWA) best matches the site's climate
    under the cold and warm scenarios: 'the warmest Holocene summers here resembled today's at ...'."""
    files = find_climate_files(a.climate_dir, a.tpat, a.ppat)
    sc = derive_scenarios(OUT / "site_variability_bounds.csv") if (OUT / "site_variability_bounds.csv").exists() else None
    d_cold = a.d_cold if a.d_cold is not None else (sc["cold"] if sc else None)
    d_warm = a.d_warm if a.d_warm is not None else (sc["warm"] if sc else None)
    if d_cold is None or d_warm is None:
        sys.exit("No scenarios: site-bounds produced no qualifying cold/warm caps; pass --d-cold and --d-warm (deg C).")
    shape = (1 + a.winter_scale) / 2 + (1 - a.winter_scale) / 2 * np.cos((np.arange(12) - 6) / 12 * 2 * np.pi)
    T, lons, lats = _read_window(files[0], (-135.0, 30.0, -100.0, 60.0))
    LON, LAT = np.meshgrid(lons, lats)
    ok = ~np.isnan(T).any(0)
    F = np.stack([T.mean(0), T.min(0), T.max(0)])[:, ok].T            # MAT, MTCO, MTWA of every land cell
    glat, glon = LAT[ok], LON[ok]
    sd = F.std(0)
    sites = pd.read_csv(OUT / "paleo_sites_climate.csv")
    sites = sites[sites.site_id.isin(WEST_SITES) | a.all_sites]
    rows = []
    for r in sites.itertuples():
        Tm = np.array([getattr(r, f"T{m:02d}") for m in range(1, 13)], float) if hasattr(r, "T01") else None
        if Tm is None:
            # fall back to the map value at the site
            i = np.abs(lats - r.lat).argmin(); j = np.abs(lons - r.lon).argmin(); Tm = T[:, i, j]
        for name, dT in [("cold", d_cold), ("warm", d_warm)]:
            Ts = Tm + dT * shape
            tgt = np.array([Ts.mean(), Ts.min(), Ts.max()])
            dist = np.sqrt((((F - tgt) / sd) ** 2).sum(1))
            best = np.argsort(dist)[:a.top]
            for k, bi in enumerate(best):
                rows.append({"site_id": r.site_id, "scenario": name, "shift_C": round(dT, 2), "rank": k + 1,
                             "analog_lat": round(glat[bi], 2), "analog_lon": round(glon[bi], 2),
                             "analog_MAT": round(F[bi, 0], 1), "analog_MTCO": round(F[bi, 1], 1), "analog_MTWA": round(F[bi, 2], 1),
                             "target_MAT": round(tgt[0], 1), "target_MTCO": round(tgt[1], 1), "target_MTWA": round(tgt[2], 1),
                             "distance_sd": round(dist[bi], 3)})
    res = pd.DataFrame(rows); res.to_csv(OUT / "holocene_analogs.csv", index=False)
    pd.set_option("display.width", 200)
    print(res[res["rank"] == 1].to_string(index=False))
    print(f"\nwrote {OUT / 'holocene_analogs.csv'} (top {a.top} analog cells per site and scenario).")
    print("Look up analog_lat/analog_lon on any map to name the place. Temperature only; precipitation not matched.")


def _site_climate(lat, lon, files, search=3):
    """Monthly T (and P) at a point; if the cell is ocean/no-data, use the nearest valid cell within
    `search` cells (coastal towns often fall in sea cells at 2.5')."""
    import rasterio
    out = []
    for paths in [files[0]] + ([files[1]] if files[1] is not None else []):
        vals = []
        for p in paths:
            with rasterio.open(p) as r:
                row, col = r.index(lon, lat)
                w = rasterio.windows.Window(col - search, row - search, 2 * search + 1, 2 * search + 1)
                a = r.read(1, window=w, boundless=True, fill_value=np.nan).astype(float)
                if r.nodata is not None: a[a == r.nodata] = np.nan
                a[a < -1e30] = np.nan
                vals.append(a)
        out.append(np.stack(vals))
    T = out[0]; ok = ~np.isnan(T).any(0)
    if not ok.any(): sys.exit(f"no land climate within {search} cells of {lat}, {lon}")
    ii, jj = np.where(ok); c = search
    k = np.argmin((ii - c) ** 2 + (jj - c) ** 2); i, j = ii[k], jj[k]
    moved_km = 111 * np.hypot((i - c) * 2.5 / 60, (j - c) * 2.5 / 60 * np.cos(np.radians(lat)))
    Tm = T[:, i, j]; Pm = out[1][:, i, j] if len(out) > 1 else np.full(12, np.nan)
    return Tm, Pm, moved_km


def cmd_profile(a):
    """Where does a place sit inside each species' climate range?  Reports, for every variable, the
    species' range edges (q01/q05/q95/q99, species-wide and within --radius-km), the place's value,
    and the 'room' to each edge. Room on the cold side = how much colder the place could become
    (for long enough to matter) before the species would be outside its known range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = find_climate_files(a.climate_dir, a.tpat, a.ppat)
    Tm, Pm, moved = _site_climate(a.lat, a.lon, files)
    here = derived(Tm, Pm, a.lat)
    if moved > 0.5: print(f"note: {a.name} falls in a no-data (sea) cell; using nearest land cell {moved:.1f} km away")
    taxa = [t.strip() for t in a.taxa.split(";") if t.strip()]
    rows = []
    for t in taxa:
        fn = OUT / "occ_climate" / (t.replace(" ", "_") + ".csv")
        if not fn.exists():
            print(f"!! no climate file for {t}: run  fetch-gbif --only \"{t.split()[-1]}\"  then  clean-occ  and  extract-climate"); continue
        d = pd.read_csv(fn)
        loc = d[_hav_km(a.lat, a.lon, d["decimalLatitude"], d["decimalLongitude"]) <= a.radius_km]
        for v in ["MTWA", "MTCO", "MAT", "GDD5"]:
            x0 = here[v]
            for scope, dd in [("species-wide", d), (f"within {a.radius_km:g} km", loc)]:
                x = dd[v].dropna()
                if len(x) < 20: continue
                q = {p: x.quantile(p) for p in [0.01, 0.05, 0.5, 0.95, 0.99]}
                rows.append({"taxon": t, "variable": v, "scope": scope, "n": len(x), "place_value": round(x0, 2),
                             "q01": round(q[0.01], 2), "q05": round(q[0.05], 2), "median": round(q[0.5], 2),
                             "q95": round(q[0.95], 2), "q99": round(q[0.99], 2),
                             "room_colder_q01": round(x0 - q[0.01], 2), "room_colder_q05": round(x0 - q[0.05], 2),
                             "room_warmer_q95": round(q[0.95] - x0, 2), "room_warmer_q99": round(q[0.99] - x0, 2),
                             "pct_of_records_colder_than_place": round(100 * (x < x0).mean(), 1)})
    if not rows: sys.exit("nothing to report")
    r = pd.DataFrame(rows); tag = a.name.replace(" ", "_")
    r.to_csv(OUT / f"profile_{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n{a.name} ({a.lat}, {a.lon}): MTWA {here['MTWA']:.2f}  MTCO {here['MTCO']:.2f}  MAT {here['MAT']:.2f}  GDD5 {here['GDD5']:.0f}  (1970-2000)")
    print(r[["taxon", "variable", "scope", "place_value", "q01", "q05", "q95", "q99", "room_colder_q01",
             "room_colder_q05", "room_warmer_q95", "pct_of_records_colder_than_place"]].to_string(index=False))
    # figure: one panel per variable, bars = q01-q99 (thin) and q05-q95 (thick), line = place
    vars_ = ["MTWA", "MTCO", "MAT"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 1.2 + 0.55 * len(taxa) * 2), constrained_layout=True)
    for ax, v in zip(axs, vars_):
        sub = r[r.variable == v].reset_index(drop=True)
        ylab = []
        for k, rr in sub.iterrows():
            y = len(sub) - 1 - k
            ax.plot([rr.q01, rr.q99], [y, y], color="0.6", lw=2)
            ax.plot([rr.q05, rr.q95], [y, y], color="tab:green" if rr.scope.startswith("species") else "tab:olive", lw=7, solid_capstyle="butt")
            ylab.append(f"{rr.taxon} ({'all' if rr.scope.startswith('species') else 'local'})")
        ax.axvline(here[v], color="tab:red", lw=1.5, label=f"{a.name} today")
        for dT, ls in [(-2.5, ":"), (2.5, ":")]:
            ax.axvline(here[v] + (dT if v != "GDD5" else 0), color="tab:red", lw=0.8, ls=ls)
        ax.set_yticks(range(len(sub))); ax.set_yticklabels(ylab[::-1], fontsize=8)
        ax.set_title({"MTWA": "Warmest month (\u00b0C)", "MTCO": "Coldest month (\u00b0C)", "MAT": "Mean annual (\u00b0C)"}[v], fontsize=10)
    fig.legend(handles=[axs[0].lines[-3]], labels=[f"{a.name} today (dotted: \u00b12.5 \u00b0C)"], loc="lower center",
               ncol=1, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{a.name}: position within species climate ranges (thick = 5-95%, thin = 1-99%; dotted = \u00b12.5 \u00b0C)", fontsize=10)
    fig.savefig(OUT / f"profile_{tag}.png", dpi=170); plt.close(fig)
    print(f"\nwrote {OUT / f'profile_{tag}.csv'} and {OUT / f'profile_{tag}.png'}")


# ----------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd")
    sp.add_parser("selftest")
    g = sp.add_parser("fetch-gbif", help="latitude-stratified random sample from GBIF")
    g.add_argument("--per-band", type=int, default=150, help="max records per 1-degree latitude band")
    g.add_argument("--chunk", type=int, default=50, help="records per random page within a band")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--max-uncertainty-m", type=float, default=5000)
    g.add_argument("--refresh", action="store_true")
    g.add_argument("--only", default="", help="download only taxa whose name contains this text")
    c = sp.add_parser("check-occ", help="coverage and truncation report for downloaded occurrences")
    k = sp.add_parser("clean-occ", help="drop spatially isolated records (planted / mis-georeferenced)")
    k.add_argument("--k", type=int, default=5, help="neighbour rank used for isolation test")
    k.add_argument("--max-km", type=float, default=200, help="drop if k-th neighbour is farther than this")
    for p in (g, c):
        p.add_argument("--lat-min", type=float, default=15); p.add_argument("--lat-max", type=float, default=72)
        p.add_argument("--lon-min", type=float, default=-170); p.add_argument("--lon-max", type=float, default=-52)
    fc = sp.add_parser("fetch-climate", help="download WorldClim 2.1 monthly tavg + prec")
    fc.add_argument("--res", default="2.5m", choices=["10m", "5m", "2.5m", "30s"],
                    help="10m~18 km, 5m~9 km, 2.5m~4.5 km, 30s~1 km (30s files are very large)")
    fc.add_argument("--climate-dir", default="climate")
    fc.add_argument("--keep-zip", action="store_true")
    e = sp.add_parser("extract-climate")
    e.add_argument("--climate-dir", default="climate")
    e.add_argument("--tpat", default="wc2.1_2.5m_tavg_{m:02d}.tif"); e.add_argument("--ppat", default="wc2.1_2.5m_prec_{m:02d}.tif")
    b = sp.add_parser("breadth"); b.add_argument("--min-n", type=int, default=30)
    b.add_argument("--n-perm", type=int, default=10000)
    s = sp.add_parser("site-bounds"); s.add_argument("--include-low-confidence", action="store_true")
    s.add_argument("--q-wide", type=float, default=0.01, help="quantile for persistent/lost/gained envelopes")
    s.add_argument("--q-narrow", type=float, default=0.05, help="quantile for never_present envelopes")
    s.add_argument("--no-geo-check", action="store_true")
    s.add_argument("--climate-dir", default="climate")
    s.add_argument("--tpat", default="wc2.1_2.5m_tavg_{m:02d}.tif"); s.add_argument("--ppat", default="wc2.1_2.5m_prec_{m:02d}.tif")
    s.add_argument("--regional-km", type=float, default=800, help="radius for regional envelopes")
    s.add_argument("--min-regional-n", type=int, default=30)
    for nm, hlp in [("holocene-maps", "three-panel maps (coldest bound / present / warmest bound) + Koppen remap"),
                    ("analogs", "present-day places whose temperatures match each site's Holocene bounds")]:
        h = sp.add_parser(nm, help=hlp)
        h.add_argument("--d-cold", type=float, default=None, help="override cold shift, deg C (negative)")
        h.add_argument("--d-warm", type=float, default=None, help="override warm shift, deg C")
        h.add_argument("--winter-scale", type=float, default=1.0,
                       help="January anomaly as a fraction of the July anomaly (1 = uniform shift)")
        h.add_argument("--climate-dir", default="climate")
        h.add_argument("--tpat", default="wc2.1_2.5m_tavg_{m:02d}.tif"); h.add_argument("--ppat", default="wc2.1_2.5m_prec_{m:02d}.tif")
        if nm == "holocene-maps":
            h.add_argument("--cd", type=float, default=0.0, help="C|D threshold: 0 (Peel 2007) or -3 (Koppen 1936)")
        else:
            h.add_argument("--top", type=int, default=5); h.add_argument("--all-sites", action="store_true")
    pr = sp.add_parser("profile", help="where a place sits inside each species' climate range")
    pr.add_argument("--taxa", default="Quercus garryana;Arbutus menziesii;Rhododendron macrophyllum")
    pr.add_argument("--lat", type=float, default=48.12); pr.add_argument("--lon", type=float, default=-123.43)
    pr.add_argument("--name", default="Port Angeles"); pr.add_argument("--radius-km", type=float, default=300)
    pr.add_argument("--climate-dir", default="climate")
    pr.add_argument("--tpat", default="wc2.1_2.5m_tavg_{m:02d}.tif"); pr.add_argument("--ppat", default="wc2.1_2.5m_prec_{m:02d}.tif")
    a = ap.parse_args()
    if a.cmd is None:
        ap.print_help()
        print(f"\nproject folder: {ROOT}\n  data:   {DATA}  (exists: {DATA.is_dir()})\n  output: {OUT}")
        return 0
    return {"selftest": cmd_selftest, "fetch-gbif": cmd_fetch_gbif, "check-occ": cmd_check_occ, "profile": cmd_profile, "holocene-maps": cmd_holocene_maps, "analogs": cmd_analogs, "fetch-climate": cmd_fetch_climate, "clean-occ": cmd_clean_occ, "extract-climate": cmd_extract,
            "breadth": cmd_breadth, "site-bounds": cmd_site_bounds}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
