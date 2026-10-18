#!/usr/bin/env python3
"""
project.py -- set up and run the whole v0.3 project from one folder.

  python code/project.py setup --old ..\\holocene_proxy     # one time: build folders, move/copy data
  python code/project.py run                                # pipeline -> results/ -> tables/ + figures/
  python code/project.py run --from site-bounds             # resume from a step
  python code/project.py run --paper-only                   # only refresh tables/ and figures/ from results/

Final layout (everything relative to the project folder, e.g. Downloads\\holocene_warmth_v03):

  main.tex, references.bib, README.md      the paper (upload to Overleaf with tables/ and figures/)
  tables/      generated LaTeX tables          figures/   generated charts
  results/     snapshot of CSVs the paper uses
  code/        build_variability_proxy.py, make_tables.py, make_figures.py, project.py
  data/        indicator_taxa.csv, taxon_site_status.csv, paleo_sites.csv, modern_reference_localities.csv,
               occurrences/ (GBIF downloads), occurrences_clean/ (after isolation filter)
  climate/     24 WorldClim 2.1 monthly GeoTIFFs (tavg + prec)
  output/      everything the pipeline writes
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
PIPE = CODE / "build_variability_proxy.py"

REFERENCE_LOCALITIES = """locality,side,lat,lon,elev_m_approx,notes
Forks,west,47.95,-124.39,110,hypermaritime
Aberdeen,west,46.98,-123.82,10,outer coast
Port Angeles,west,48.12,-123.43,30,partial rainshadow
Sequim,west,48.08,-123.10,60,rainshadow core
Friday Harbor,west,48.53,-123.02,30,San Juan rainshadow
Bellingham,west,48.75,-122.48,30,
Seattle,west,47.61,-122.33,50,
Olympia,west,47.04,-122.90,60,S Puget prairies
Vancouver WA,west,45.64,-122.66,60,lower Columbia
Wenatchee,east,47.42,-120.31,200,
Ellensburg,east,46.99,-120.55,450,forest-steppe approach
Yakima,east,46.60,-120.51,325,
Richland,east,46.28,-119.28,120,driest lowland
Moses Lake,east,47.13,-119.28,325,
Walla Walla,east,46.06,-118.34,290,hot canyon margin
Omak,east,48.41,-119.53,260,Okanogan
Spokane,east,47.66,-117.43,600,forest-steppe
Pullman,east,46.73,-117.18,770,Palouse
"""

# pipeline steps, in order (name, arguments)
STEPS = [
    ("fetch-rhododendron", ["fetch-gbif", "--only", "macrophyllum"]),
    ("clean-occ", ["clean-occ"]),
    ("check-occ", ["check-occ"]),
    ("extract-climate", ["extract-climate"]),
    ("breadth", ["breadth"]),
    ("site-bounds", ["site-bounds"]),
    ("holocene-maps", ["holocene-maps"]),
    ("analogs", ["analogs"]),
    ("profile", ["profile"]),
]
# files the paper reads, and where they come from
SYNC_FROM_OUTPUT = ["niche_breadth.csv", "niche_breadth_by_group.csv", "niche_breadth_permutation.csv",
                    "site_variability_bounds.csv", "paleo_sites_climate.csv", "occurrence_coverage.csv",
                    "dropped_isolated_records.csv", "holocene_koppen_change.csv", "holocene_koppen_area.csv",
                    "holocene_analogs.csv", "profile_Port_Angeles.csv"]
SYNC_FROM_DATA = ["indicator_taxa.csv", "taxon_site_status.csv", "paleo_sites.csv"]
MAP_FIGURES = {"holocene_map_MTWA.png": "fig_map_mtwa.png", "holocene_map_koppen.png": "fig_map_koppen.png"}


def say(msg): print(msg, flush=True)


# ----------------------------------------------------------------------------- setup
def setup(a):
    for d in ["data", "output", "figures", "climate", "results", "tables"]:
        (ROOT / d).mkdir(exist_ok=True)
    say(f"project folder: {ROOT}")

    # 1. climate GeoTIFFs -> ROOT/climate (from code/climate, any subfolder, or the old project)
    sources = [CODE / "climate"]
    if a.old: sources.append(Path(a.old).expanduser().resolve() / "climate")
    moved = 0
    for src in sources:
        if not src.is_dir(): continue
        for tif in sorted(src.rglob("*.tif")):
            dest = ROOT / "climate" / tif.name
            if not dest.exists():
                shutil.copy2(tif, dest) if src != CODE / "climate" else shutil.move(str(tif), dest)
                moved += 1
    n_t = len(list((ROOT / "climate").glob("*tavg_*.tif"))); n_p = len(list((ROOT / "climate").glob("*prec_*.tif")))
    say(f"climate/: {n_t} temperature + {n_p} precipitation files ({moved} added)")
    # remove code/climate if everything in it now exists in ROOT/climate
    cc = CODE / "climate"
    if cc.is_dir():
        leftovers = [p for p in cc.rglob("*.tif") if not (ROOT / "climate" / p.name).exists()]
        if not leftovers:
            shutil.rmtree(cc); say("removed code/climate (duplicates only)")
        else:
            say(f"kept code/climate: {len(leftovers)} files not found in climate/")
    if n_t < 12 or n_p < 12:
        say("!! climate/ is incomplete; run:  python code/build_variability_proxy.py fetch-climate")

    # 2. occurrences from the old project
    if a.old:
        old = Path(a.old).expanduser().resolve()
        if not old.is_dir(): sys.exit(f"--old folder not found: {old}")
        for sub in ["occurrences", "occurrences_clean"]:
            s = old / "data" / sub
            if s.is_dir():
                shutil.copytree(s, ROOT / "data" / sub, dirs_exist_ok=True)
                say(f"data/{sub}/: {len(list((ROOT / 'data' / sub).glob('*.csv')))} species files")
            else:
                say(f"(no {s})")
    # 3. data tables: newest versions travel in results/; never overwrite edited copies in data/
    for name in SYNC_FROM_DATA:
        dest = ROOT / "data" / name
        if dest.exists(): say(f"data/{name}: kept existing"); continue
        for src in [ROOT / "results" / name, (Path(a.old) / "data" / name) if a.old else None]:
            if src and src.exists():
                shutil.copy2(src, dest); say(f"data/{name}: copied from {src.parent.name}/"); break
        else:
            say(f"!! data/{name} missing")
    ref = ROOT / "data" / "modern_reference_localities.csv"
    if not ref.exists():
        ref.write_text(REFERENCE_LOCALITIES, encoding="utf-8"); say("data/modern_reference_localities.csv: written")
    say("\nsetup done. Next:  python code/project.py run")


# ----------------------------------------------------------------------------- run
OPTIONAL = {"holocene-maps", "analogs", "profile"}      # a failure here warns but does not stop the run
WARNINGS = []


def run_step(name, args):
    cmd = [sys.executable, str(PIPE)] + args
    say(f"\n=== {' '.join(args)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        if name in OPTIONAL:
            WARNINGS.append(name); say(f"!! {name} did not complete; continuing (see message above)")
        else:
            sys.exit(f"step failed: {' '.join(args)}  (fix, then resume with:  python code/project.py run --from {name})")


def sync():
    say("\n=== sync output/ and data/ -> results/, maps -> figures/")
    for name in SYNC_FROM_OUTPUT:
        s = ROOT / "output" / name
        if s.exists(): shutil.copy2(s, ROOT / "results" / name); say(f"  results/{name}")
        else: say(f"  (not produced: {name})")
    for name in SYNC_FROM_DATA:
        s = ROOT / "data" / name
        if s.exists(): shutil.copy2(s, ROOT / "results" / name)
    for src, dst in MAP_FIGURES.items():
        s = ROOT / "output" / src
        if s.exists(): shutil.copy2(s, ROOT / "figures" / dst); say(f"  figures/{dst}")


def paper():
    for script in ["make_tables.py", "make_figures.py"]:
        say(f"\n=== {script}")
        r = subprocess.run([sys.executable, str(CODE / script)], cwd=ROOT)
        if r.returncode != 0: sys.exit(f"{script} failed")


def run(a):
    if not PIPE.exists(): sys.exit(f"missing {PIPE}")
    if not a.paper_only:
        names = [n for n, _ in STEPS]
        start = 0
        if a.from_step:
            if a.from_step not in names: sys.exit(f"--from must be one of: {', '.join(names)}")
            start = names.index(a.from_step)
        for name, args in STEPS[start:]:
            if name == "fetch-rhododendron" and a.no_fetch: say("\n=== fetch skipped (--no-fetch)"); continue
            run_step(name, args)
        sync()
    paper()
    if WARNINGS: say(f"\nfinished with warnings in: {', '.join(WARNINGS)} (their tables/figures stay as placeholders)")
    say("\nall done. Upload to Overleaf: main.tex, references.bib, tables/, figures/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd")
    s = sp.add_parser("setup"); s.add_argument("--old", default=None, help="old pipeline folder, e.g. ..\\holocene_proxy")
    r = sp.add_parser("run")
    r.add_argument("--from", dest="from_step", default=None)
    r.add_argument("--paper-only", action="store_true", help="only rebuild tables/ and figures/ from results/")
    r.add_argument("--no-fetch", action="store_true", help="skip the GBIF download step")
    a = ap.parse_args()
    if a.cmd == "setup": setup(a)
    elif a.cmd == "run": run(a)
    else: ap.print_help()


if __name__ == "__main__":
    main()
