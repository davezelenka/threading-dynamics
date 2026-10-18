#!/usr/bin/env python3
"""
make_tables.py -- write every data table in main.tex from the result CSVs.

    python code/make_tables.py                 # reads results/, writes tables/*.tex
    python code/make_tables.py --results PATH  # e.g. the pipeline's output/ folder

Rerun after any new pipeline run and re-upload tables/ to Overleaf; main.tex \\input{}s these files,
so the paper's numbers always match the data. No numbers in the tables are typed by hand.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEST = ["BGL", "LWA", "KIL", "PEN", "HOH"]
EAST = ["WIL", "WCL", "CAR"]
SITE_NAME = {"BGL": "Battle Ground Lake", "LWA": "Lake Washington", "KIL": "Killebrew Lake Fen",
             "PEN": "Pender Is.\\ (BC)", "HOH": "Hoh valley", "WIL": "Williams Lake Fen",
             "WCL": "Wildcat Lake", "CAR": "Carp Lake", "CRO": "Crocker Lake", "WEN": "Wentworth Lake",
             "DAV": "Davis Lake", "CAS": "Castor Lake"}
VAR_TEX = {"MTWA": "MTWA", "MTCO": "MTCO", "MAT": "MAT", "GDD5": "GDD5"}


def it(name):
    """Italicise a binomial; leave group names plain."""
    if not isinstance(name, str) or not name: return "--"
    return "\\emph{" + name + "}"


def short(name):
    if not isinstance(name, str) or " " not in name: return it(name)
    g, s = name.split(" ", 1)
    return "\\emph{" + g[0] + ".\\ " + s + "}"


def is_true(x):
    return (not pd.isna(x)) and str(x).strip().lower() in ("true", "1", "1.0")


def status(row, side):
    """ok = set by a taxon limited by this variable; prov = provisional; GEO = edge of available climate."""
    if pd.isna(row[f"max_{side}"]): return "--"
    if is_true(row.get(f"{side}_geo_truncated")): return "GEO"
    return "ok" if is_true(row.get(f"{side}_set_by_limiting_taxon")) else "prov"


def write(name, body):
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables" / name).write_text(body, encoding="utf-8")
    print(f"wrote tables/{name}")


def longtable(colspec, header, rows, caption, label, note=None, size="small"):
    h = " & ".join(header) + " \\\\"
    out = [f"\\begin{{longtable}}{{{colspec}}}",
           f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule", h, "\\midrule", "\\endfirsthead",
           "\\toprule", h, "\\midrule", "\\endhead", "\\bottomrule"]
    if note: out.append(f"\\multicolumn{{{len(header)}}}{{p{{0.95\\textwidth}}}}{{\\footnotesize {note}}}\\\\")
    out.append("\\endfoot")
    out += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    out.append("\\end{longtable}")
    return "{\\" + size + "\n" + "\n".join(out) + "\n}\n"


def f(x, d=1):
    if pd.isna(x): return "--"
    x = round(float(x), d)
    return f"{(0.0 if x == 0 else x):.{d}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(ROOT / "results"))
    R = Path(ap.parse_args().results)
    taxa = pd.read_csv(R / "indicator_taxa.csv")
    cov = pd.read_csv(R / "occurrence_coverage.csv")
    nb = pd.read_csv(R / "niche_breadth.csv")
    grp = pd.read_csv(R / "niche_breadth_by_group.csv")
    perm = pd.read_csv(R / "niche_breadth_permutation.csv")
    sb = pd.read_csv(R / "site_variability_bounds.csv")
    sc = pd.read_csv(R / "paleo_sites_climate.csv")
    st = pd.read_csv(R / "taxon_site_status.csv")
    try:
        drop = pd.read_csv(R / "dropped_isolated_records.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        drop = pd.DataFrame(columns=["taxon", "lat", "lon"])

    # ---------------- T: occurrence data per taxon
    c = cov.set_index("taxon")
    d = drop.groupby("taxon").size()
    env = nb[nb.variable == "MTWA"].set_index("taxon")
    rows = []
    for r in taxa.itertuples():
        t = r.gbif_name
        if t not in c.index: rows.append([it(t), r.group.replace("_", " "), "not downloaded", "", "", "", ""]); continue
        rows.append([it(t), r.group.replace("_", " ").replace("reference", "ref."), int(c.loc[t, "n"]), int(d.get(t, 0)),
                     f"{c.loc[t, 'lat_q05']:.1f}--{c.loc[t, 'lat_q95']:.1f}",
                     f"{env.loc[t, 'L05']:.1f}--{env.loc[t, 'U95']:.1f}" if t in env.index else "--",
                     r.variables_bounded.replace("_", "\\_").replace("(not temperature)", "(not T)")])
    write("tab_taxa.tex", longtable(
        ">{\\raggedright\\arraybackslash}p{0.21\\textwidth}>{\\raggedright\\arraybackslash}p{0.10\\textwidth}rr"
        ">{\\raggedright\\arraybackslash}p{0.09\\textwidth}>{\\raggedright\\arraybackslash}p{0.10\\textwidth}"
        ">{\\raggedright\\arraybackslash}p{0.16\\textwidth}",
        ["\\textbf{Taxon}", "\\textbf{Group}", "\\textbf{n}", "\\textbf{Drop.}", "\\textbf{Lat.\\ 5--95\\%}",
         "\\textbf{MTWA 5--95\\% (\\degC)}", "\\textbf{Thought limited by}"], rows,
        "Indicator taxa and occurrence data. n: cleaned, climate-annotated records after latitude-stratified sampling; "
        "Drop.: spatially isolated records removed (likely planted or mis-georeferenced). Last column: variables the taxon "
        "is thought to be limited by (used to grade bounds as \\emph{ok} or \\emph{provisional}).", "tab:taxa"))

    # ---------------- T: site climate
    sc2 = sc.set_index("site_id")
    rows = []
    for sid in WEST + ["CRO", "WEN"] + EAST + ["CAS"]:
        if sid not in sc2.index: continue
        s = sc2.loc[sid]
        rows.append([SITE_NAME.get(sid, sid), "W" if sid in WEST + ["CRO", "WEN"] else "E", f"{s.lat:.2f}", f"{s.lon:.2f}",
                     f(s.MAT), f(s.MTCO), f(s.MTWA), f(s.GDD5, 0)])
    write("tab_sites.tex", longtable(
        ">{\\raggedright\\arraybackslash}p{0.22\\textwidth}crrrrrr",
        ["\\textbf{Site}", "\\textbf{Side}", "\\textbf{Lat}", "\\textbf{Lon}", "\\textbf{MAT}", "\\textbf{MTCO}",
         "\\textbf{MTWA}", "\\textbf{GDD5}"], rows,
        "Modern (1970--2000) climate at the sediment sites, WorldClim 2.1 at 2.5$'$. Temperatures in \\degC. "
        "Coordinates for Hoh and Wentworth are approximate and place those sites too high and cold.", "tab:sites"))

    # ---------------- T: taxon-site status
    rows = []
    for r in st.itertuples():
        rows.append([SITE_NAME.get(r.site_id, r.site_id), it(r.gbif_name), r.status.replace("_", " "),
                     r.confidence, ("excluded" if str(r.dispersal_limited_risk).upper().startswith("HIGH") else "used")])
    write("tab_status.tex", longtable(
        ">{\\raggedright\\arraybackslash}p{0.20\\textwidth}>{\\raggedright\\arraybackslash}p{0.25\\textwidth}"
        ">{\\raggedright\\arraybackslash}p{0.16\\textwidth}>{\\raggedright\\arraybackslash}p{0.14\\textwidth}l",
        ["\\textbf{Site}", "\\textbf{Taxon}", "\\textbf{Holocene status}", "\\textbf{Confidence}", "\\textbf{In bounds}"],
        rows, "Holocene status of taxa at the sites, compiled from the sediment literature (sources in "
        "\\texttt{taxon\\_site\\_status.csv}). Only \\emph{persistent} and \\emph{never present} enter upper bounds; "
        "\\emph{lost} and \\emph{gained} would give lower bounds; \\emph{abundance change} gives no hard bound.", "tab:status"))

    # ---------------- T: site bounds (MTWA and MAT)
    rows = []
    for sid in WEST + EAST:
        for scope in ["regional", "species"]:
            g = sb[(sb.site_id == sid) & (sb.scope == scope)].set_index("variable")
            if g.empty or g["max_up"].isna().all(): continue
            cells = [SITE_NAME[sid] if scope == "regional" else "", scope]
            for v in ["MTWA", "MAT"]:
                if v not in g.index: cells += ["--", "--"]; continue
                r = g.loc[v]
                cells.append(f"$+${f(r.max_up)} \\textsuperscript{{{status(r, 'up')}}}")
                cells.append(f"$-${f(r.max_down)} \\textsuperscript{{{status(r, 'down')}}}")
            g2 = g.loc["MTWA"] if "MTWA" in g.index else None
            cells.append(short(g2.binding_up) if g2 is not None else "--")
            cells.append(short(g2.binding_down) if g2 is not None else "--")
            rows.append(cells)
    write("tab_bounds.tex", longtable(
        ">{\\raggedright\\arraybackslash}p{0.12\\textwidth}lrrrr>{\\raggedright\\arraybackslash}p{0.11\\textwidth}"
        ">{\\raggedright\\arraybackslash}p{0.11\\textwidth}",
        ["\\textbf{Site}", "\\textbf{Envelope}", "\\textbf{MTWA up}", "\\textbf{MTWA down}", "\\textbf{MAT up}",
         "\\textbf{MAT down}", "\\textbf{Sets MTWA up}", "\\textbf{Sets MTWA down}"], rows,
        "Maximum Holocene departure from the 1970--2000 mean permitted by each site's taxa (\\degC). "
        "\\emph{regional}: envelopes from records within 800~km; \\emph{species}: whole range. Grades: "
        "\\textsuperscript{ok} set by a taxon thought to be limited by that variable; \\textsuperscript{prov} provisional "
        "(limiting taxon limited mainly by something else); \\textsuperscript{GEO} not informative (the taxon's edge is the "
        "edge of land climate available within 800~km). Carp Lake has no usable status records.", "tab:bounds", size="footnotesize"))

    # ---------------- T: statewide uniform-shift scenarios
    w = sb[(sb.variable == "MTWA") & sb.site_id.isin(WEST)]
    cold = w[w.max_down.notna() & w.down_set_by_limiting_taxon.map(is_true) & ~w.down_geo_truncated.map(is_true)]
    warm = w[w.max_up.notna() & ~w.up_geo_truncated.map(is_true)]
    rows = []
    for lab, df, col, bcol, lcol in [("Coldest permitted", cold, "max_down", "binding_down", "down_set_by_limiting_taxon"),
                                     ("Warmest permitted", warm, "max_up", "binding_up", "up_set_by_limiting_taxon")]:
        if df.empty: continue
        k = df.loc[df[col].idxmin()]
        sign = "$-$" if col == "max_down" else "$+$"
        rows.append([lab, sign + f(k[col], 2), SITE_NAME[k.site_id], k.scope, it(k[bcol]),
                     "ok" if is_true(k[lcol]) else "provisional",
                     ", ".join(f"{SITE_NAME[s]} {f(v)}" for s, v in df.groupby("site_id")[col].min().sort_values().items())])
    write("tab_scenarios.tex", longtable(
        ">{\\raggedright\\arraybackslash}p{0.11\\textwidth}r>{\\raggedright\\arraybackslash}p{0.11\\textwidth}l"
        ">{\\raggedright\\arraybackslash}p{0.12\\textwidth}l>{\\raggedright\\arraybackslash}p{0.24\\textwidth}",
        ["\\textbf{Scenario}", "\\textbf{MTWA shift}", "\\textbf{Binding site}", "\\textbf{Env.}", "\\textbf{Binding taxon}",
         "\\textbf{Grade}", "\\textbf{All western caps (smallest per site)}"], rows,
        "Uniform-shift scenarios for the western lowlands. Under a spatially uniform change, the smallest cap among "
        "sites binds. Cold: smallest downward cap graded \\emph{ok}. Warm: smallest upward cap not graded GEO.",
        "tab:scenarios", size="footnotesize"))

    # ---------------- T: group breadth medians
    rows = []
    for v in ["MTWA", "MTCO", "MAT", "GDD5"]:
        g = grp[grp.variable == v].set_index("group")
        d0 = 0 if v == "GDD5" else 2
        for gname in ["west_lowland", "east_lowland", "alpine_reference"]:
            if gname not in g.index: continue
            r = g.loc[gname]
            rows.append([VAR_TEX[v] if gname == "west_lowland" else "", gname.replace("_", " ").replace("reference", "ref."),
                         f(r.breadth, d0), f(r.local_breadth, d0), f(r.breadth_resid, d0)])
    write("tab_breadth.tex", longtable(
        "llrrr", ["\\textbf{Variable}", "\\textbf{Group}", "\\textbf{Breadth}", "\\textbf{Local breadth}",
                  "\\textbf{Range-size residual}"], rows,
        "Median climatic niche breadth (5th--95th percentile spread) by group. Local breadth: median within 1$^\\circ$ "
        "latitude bands. Residual: breadth after regression on log range size \\citep{slatyer2013}. "
        "Labrador tea and disturbance taxa excluded.", "tab:breadth"))

    # ---------------- T: permutation tests
    rows = []
    for r in perm.itertuples():
        rows.append([r.variable, r.measure.replace("_", " "), r.contrast.replace("_lowland", "").replace("_reference", "")
                     .replace(" - ", " $-$ "), f"{r.n1}/{r.n2}", f(r.median_diff, 0 if r.variable == "GDD5" else 2),
                     ("\\textbf{" + f"{r.p_perm:.4f}" + "}") if r.p_perm < 0.01 else f"{r.p_perm:.4f}"])
    write("tab_permutation.tex", longtable(
        "lllrrr", ["\\textbf{Var.}", "\\textbf{Measure}", "\\textbf{Contrast}", "\\textbf{n}", "\\textbf{Median diff.}",
                   "\\textbf{p}"], rows,
        f"Permutation tests of group differences in median breadth ({len(perm)} tests, 10,000 label shuffles each). "
        f"Bold: $p<0.01$. Strict Bonferroni threshold: {0.05 / len(perm):.4f}.", "tab:perm"))

    # ---------------- T: oak and madrone at the rainshadow sites
    env_all = nb.set_index(["taxon", "variable"])
    rows = []
    for t in ["Quercus garryana", "Arbutus menziesii", "Rhododendron macrophyllum"]:
        for v in ["MTWA", "MTCO", "GDD5"]:
            if (t, v) not in env_all.index:
                if v == "MTWA": rows.append([it(t), "\\multicolumn{5}{l}{not yet downloaded (run fetch-gbif --only macrophyllum)}"])
                continue
            L, U = env_all.loc[(t, v), "L05"], env_all.loc[(t, v), "U95"]
            d0 = 0 if v == "GDD5" else 1
            cells = [it(t) if v == "MTWA" else "", v, f(L, d0) + "--" + f(U, d0)]
            for sid in ["KIL", "PEN", "CRO"]:
                x = sc2.loc[sid, v]
                cells.append(f"{f(x - L, d0)} / {f(U - x, d0)}")
            rows.append(cells)
    write("tab_oak_madrone.tex", longtable(
        ">{\\raggedright\\arraybackslash}p{0.22\\textwidth}lrrrr",
        ["\\textbf{Taxon}", "\\textbf{Var.}", "\\textbf{5--95\\% range}", "\\textbf{Killebrew}", "\\textbf{Pender}",
         "\\textbf{Crocker}"], rows,
        "Room for cooling / room for warming (\\degC; GDD5 in degree-days) before the north Olympic and San Juan "
        "rainshadow sites leave the 5--95\\% range of each species. Negative room: the site already lies beyond "
        "the species' 5th percentile.", "tab:oakmadrone"))
    koppen_table(R)


def koppen_table(R):
    """Optional: area changing thermal-Koppen class, from build_variability_proxy.py holocene-maps."""
    fc, fa = R / "holocene_koppen_change.csv", R / "holocene_koppen_area.csv"
    KOP = "K\\\"{o}ppen"
    if not fc.exists():
        write("tab_koppen.tex",
              "\\begin{table}[htbp]\\caption{Area changing thermal K\\\"{o}ppen class at each bound (placeholder).}"
              "\\label{tab:koppen}\n"
              "\\begin{center}\\fbox{\\parbox{0.9\\textwidth}{\\centering\\small Table placeholder: run "
              "\\texttt{build\\_variability\\_proxy.py holocene-maps}, copy \\texttt{holocene\\_koppen\\_change.csv} and "
              "\\texttt{holocene\\_koppen\\_area.csv} into \\texttt{results/}, then rerun \\texttt{make\\_tables.py}.}}\\end{center}\\end{table}\n")
        return
    ch = pd.read_csv(fc); ar = pd.read_csv(fa)
    full = ar["class"].astype(str).str.match(r"^(B|[CD][sfw])").any()
    kind = (f"{KOP}--Geiger class (precipitation held at its present values)" if full else
            f"thermal {KOP} class (precipitation letter and B class not determined)")
    rows = [[r.scenario.replace("(bound)", "").strip(), r.region, f(r.pct_area_changing_class)] for r in ch.itertuples()]
    body = longtable("llr", ["\\textbf{Scenario}", "\\textbf{Region}", "\\textbf{\\% area changing class}"], rows,
                     f"Share of area whose {kind} differs from the present under each temperature bound.", "tab:koppen")
    piv = ar[ar.region == "statewide"].pivot_table(index="class", columns="scenario", values="area_pct", fill_value=0)
    cols = [c for c in piv.columns if c.startswith("Cold")] + [c for c in piv.columns if c.startswith("Present")] + \
           [c for c in piv.columns if c.startswith("Warm")]
    rows2 = [[k] + [f(piv.loc[k, c]) for c in cols] for k in piv.index]
    body += longtable("l" + "r" * len(cols), ["\\textbf{Class}"] + ["\\textbf{" + c.split(" (")[0] + "}" for c in cols],
                      rows2, f"Statewide area (\\%) in each {kind.split(' (')[0]}.", "tab:koppenarea")
    write("tab_koppen.tex", body)


if __name__ == "__main__":
    main()
