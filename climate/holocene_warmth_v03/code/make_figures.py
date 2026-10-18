#!/usr/bin/env python3
"""
make_figures.py -- data figures for main.tex, from the same result CSVs as make_tables.py.

    python code/make_figures.py                 # reads results/, writes figures/*.png
    python code/make_figures.py --results PATH

Upload figures/ to Overleaf: main.tex fills each placeholder automatically when the file exists.
The two map figures (fig_map_mtwa.png, fig_map_koppen.png) come from the pipeline command
`build_variability_proxy.py holocene-maps` (it needs the WorldClim rasters); copy
output/holocene_map_MTWA.png -> figures/fig_map_mtwa.png and output/holocene_map_koppen.png -> figures/fig_map_koppen.png.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
WEST = ["BGL", "LWA", "KIL", "PEN", "HOH"]
EAST = ["WIL", "WCL"]
NAME = {"BGL": "Battle Ground L.", "LWA": "Lake Washington", "KIL": "Killebrew L. Fen", "PEN": "Pender Is. (BC)",
        "HOH": "Hoh valley*", "WIL": "Williams L. Fen", "WCL": "Wildcat L.", "CAR": "Carp L.", "CRO": "Crocker L.",
        "WEN": "Wentworth L.*", "DAV": "Davis L.", "CAS": "Castor L."}
GRADE_COL = {"ok": "#1b7837", "prov": "#d6604d", "GEO": "#bababa"}
WA_OUTLINE = [(-124.75, 48.40), (-123.25, 48.22), (-123.12, 48.42), (-123.25, 48.70), (-123.02, 48.83),
              (-123.32, 49.00), (-117.03, 49.00), (-117.04, 46.42), (-116.92, 46.00), (-118.98, 46.00),
              (-119.60, 45.93), (-120.20, 45.72), (-120.90, 45.65), (-121.20, 45.60), (-121.90, 45.65),
              (-122.25, 45.55), (-122.76, 45.65), (-122.80, 45.95), (-123.20, 46.18), (-123.90, 46.25),
              (-124.20, 46.40), (-124.20, 47.00), (-124.50, 47.60), (-124.75, 48.10), (-124.75, 48.40)]
CREST = [(-121.05, 49.00), (-121.15, 48.50), (-121.30, 47.80), (-121.40, 47.20), (-121.50, 46.70),
         (-121.60, 46.20), (-121.85, 45.70)]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def is_true(x): return (not pd.isna(x)) and str(x).strip().lower() in ("true", "1", "1.0")


def grade(r, side):
    if pd.isna(r[f"max_{side}"]): return None
    if is_true(r.get(f"{side}_geo_truncated")): return "GEO"
    return "ok" if is_true(r.get(f"{side}_set_by_limiting_taxon")) else "prov"


def save(fig, name):
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / name, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote figures/{name}")


def fig_sites(sc, st):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    o = np.array(WA_OUTLINE); c = np.array(CREST)
    ax.plot(o[:, 0], o[:, 1], "k", lw=0.8); ax.plot(c[:, 0], c[:, 1], "k--", lw=0.8, label="Cascade crest (approx.)")
    n_status = st.groupby("site_id").size()
    for r in sc.itertuples():
        col = "#2166ac" if r.lon < -121.5 else "#b2182b"
        ax.scatter(r.lon, r.lat, s=25 + 12 * n_status.get(r.site_id, 0), color=col, edgecolor="k", zorder=3)
        ax.annotate(NAME.get(r.site_id, r.site_id), (r.lon, r.lat), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.scatter([], [], color="#2166ac", edgecolor="k", label="western site"); ax.scatter([], [], color="#b2182b", edgecolor="k", label="eastern site")
    ax.set_xlim(-125.2, -116.6); ax.set_ylim(45.4, 49.3); ax.set_aspect(1 / np.cos(np.radians(47.3)))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(fontsize=7, loc="lower left", frameon=False)
    ax.set_title("Sediment records (marker size = number of taxon status records)", fontsize=9)
    save(fig, "fig_sites.png")


def fig_bounds(sb):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    sites = [s for s in WEST + EAST if s in set(sb.site_id)]
    for ax, v, lab in zip(axs, ["MTWA", "MAT"], ["Warmest-month mean", "Mean annual temperature"]):
        for k, sid in enumerate(sites):
            for dy, scope in [(0.15, "regional"), (-0.15, "species")]:
                g = sb[(sb.site_id == sid) & (sb.scope == scope) & (sb.variable == v)]
                if g.empty: continue
                r = g.iloc[0]; y = len(sites) - 1 - k + dy
                for side, sgn in [("up", 1), ("down", -1)]:
                    gr = grade(r, side)
                    if gr is None: continue
                    ax.plot([0, sgn * r[f"max_{side}"]], [y, y], color=GRADE_COL[gr], lw=4 if scope == "regional" else 1.6,
                            solid_capstyle="butt")
        ax.axvline(0, color="k", lw=0.8)
        for x in (-2.5, 2.5): ax.axvline(x, color="k", lw=0.6, ls=":")
        ax.set_xlim(-9, 9); ax.set_xlabel(f"{v}: permitted departure (\u00b0C vs 1970-2000)")
    axs[0].set_yticks(range(len(sites))); axs[0].set_yticklabels([NAME[s] for s in sites][::-1])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=GRADE_COL[g], lw=4) for g in ["ok", "prov", "GEO"]] + \
        [Line2D([], [], color="k", lw=4), Line2D([], [], color="k", lw=1.6), Line2D([], [], color="k", lw=0.6, ls=":")]
    fig.legend(h, ["ok: limiting taxon", "provisional", "GEO: not informative", "regional envelope (thick)",
                   "species-wide envelope (thin)", "\u00b12.5 \u00b0C"], loc="lower center", ncol=6, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Upper (right) and lower (left) caps on Holocene temperature at each site. "
                 "*approximate coordinates", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97)); save(fig, "fig_bounds.png")


def fig_envelopes(nb, sc):
    taxa = ["Tsuga mertensiana", "Picea sitchensis", "Tsuga heterophylla", "Pseudotsuga menziesii",
            "Arbutus menziesii", "Quercus garryana", "Artemisia tridentata", "Celtis reticulata"]
    taxa = [t for t in taxa if t in set(nb.taxon)]
    e = nb.set_index(["taxon", "variable"])
    s = sc.set_index("site_id")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, v, lab in zip(axs, ["MTWA", "MTCO"], ["Warmest-month mean (\u00b0C)", "Coldest-month mean (\u00b0C)"]):
        for k, t in enumerate(taxa):
            y = len(taxa) - 1 - k
            L, U = e.loc[(t, v), "L05"], e.loc[(t, v), "U95"]
            ax.plot([L, U], [y, y], lw=7, color="#4d9221", solid_capstyle="butt", alpha=0.85)
        for sid, mk in [("LWA", "o"), ("BGL", "s"), ("KIL", "^"), ("WIL", "D")]:
            if sid in s.index: ax.axvline(s.loc[sid, v], color="k", lw=0.8, ls="--")
            ax.text(s.loc[sid, v], len(taxa) - 0.35, sid, rotation=90, fontsize=6, ha="right", va="bottom")
        ax.set_xlabel(lab)
    axs[0].set_yticks(range(len(taxa))); axs[0].set_yticklabels([t for t in taxa][::-1], style="italic")
    fig.suptitle("Species climate envelopes (5th-95th percentile of occurrences) and site values (dashed)", fontsize=9)
    fig.tight_layout(); save(fig, "fig_envelopes.png")


def fig_breadth(nb, taxa, perm):
    use = set(taxa[taxa.use_in_breadth.astype(str).str.lower() != "false"].gbif_name)
    d = nb[nb.taxon.isin(use) & nb.group.isin(["west_lowland", "east_lowland", "alpine_reference"])]
    order = ["alpine_reference", "west_lowland", "east_lowland"]; lab = ["alpine", "west lowland", "east lowland"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axs, [("breadth", "Breadth (whole range)"), ("local_breadth", "Local breadth (1\u00b0 bands)"),
                                     ("breadth_resid", "Residual after range size")]):
        dv = d[d.variable == "MTWA"]
        for i, g in enumerate(order):
            x = dv[dv.group == g][col].values
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(x)), x, s=18, color=["#5aae61", "#2166ac", "#b2182b"][i], alpha=0.8)
            ax.plot([i - 0.25, i + 0.25], [np.median(x)] * 2, color="k", lw=2)
        p = perm[(perm.variable == "MTWA") & (perm.measure == col) & (perm.contrast == "east_lowland - west_lowland")]
        if len(p): ax.set_title(f"{title}\neast vs west p = {p.p_perm.iloc[0]:.3f}", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(lab)
    axs[0].set_ylabel("Warmest-month breadth (\u00b0C)")
    fig.suptitle("Summer-temperature niche breadth by group (one point per taxon; bar = median)", fontsize=9)
    fig.tight_layout(); save(fig, "fig_breadth.png")


def fig_oak_madrone(nb, sc):
    e = nb.set_index(["taxon", "variable"]); s = sc.set_index("site_id")
    taxa = [t for t in ["Quercus garryana", "Arbutus menziesii", "Rhododendron macrophyllum"] if (t, "MTWA") in e.index]
    sites = ["KIL", "PEN", "CRO"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, v, lab in zip(axs, ["MTWA", "MTCO"], ["warmest month", "coldest month"]):
        ylab = []
        for k, t in enumerate(taxa):
            for j, sid in enumerate(sites):
                y = -(k * (len(sites) + 1) + j)
                x0 = s.loc[sid, v]; L, U = e.loc[(t, v), "L05"], e.loc[(t, v), "U95"]
                cool, warm = round(x0 - L, 1) + 0.0, round(U - x0, 1)
                if cool > 0: ax.barh(y, -cool, color="#4393c3", height=0.8)
                ax.barh(y, warm, color="#f4a582", height=0.8)
                ax.text(-0.15 if cool <= 0 else -cool - 0.15, y, f"{cool:+.1f}" + (" (beyond 5th pct)" if cool <= 0 else ""),
                        va="center", ha="right", fontsize=6.5, color="#b2182b" if cool <= 0 else "k")
                ax.text(warm + 0.15, y, f"{warm:.1f}", va="center", ha="left", fontsize=6.5)
                ylab.append((y, f"{t.split()[0][0]}. {t.split()[1]} @ {NAME[sid]}"))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"{v} (\u00b0C)"); ax.set_xlim(-6.5, 10)
        ax.set_yticks([y for y, _ in ylab]); ax.set_yticklabels([l for _, l in ylab], fontsize=7)
    fig.suptitle("Rainshadow sites within the 5-95% ranges of Mediterranean-affinity trees: "
                 "room to cool (blue, left) and to warm (orange, right)", fontsize=9)
    fig.tight_layout(); save(fig, "fig_oak_madrone.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(ROOT / "results"))
    R = Path(ap.parse_args().results)
    sb = pd.read_csv(R / "site_variability_bounds.csv"); nb = pd.read_csv(R / "niche_breadth.csv")
    sc = pd.read_csv(R / "paleo_sites_climate.csv"); st = pd.read_csv(R / "taxon_site_status.csv")
    taxa = pd.read_csv(R / "indicator_taxa.csv"); perm = pd.read_csv(R / "niche_breadth_permutation.csv")
    fig_sites(sc, st); fig_bounds(sb); fig_envelopes(nb, sc); fig_breadth(nb, taxa, perm); fig_oak_madrone(nb, sc)


if __name__ == "__main__":
    main()
