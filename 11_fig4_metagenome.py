"""Fig 4: metagenome community of the relief batch (A0_1 excluded). Run 10 first.

A  species-level Bray-Curtis PCoA with group hulls and centroid spiders; inset: 2 x 2 PERMANOVA variance partition
B  chord diagram linking dominant genera (left) to treatments (right), group means
C  clustered heatmap of the top genera (row z-scores of log10 abundance) with sample dendrogram and Shannon track
D  salt response of each genus without Ni (x) against with Ni (y), log2 fold change

Panels are placed on a common grid in inches: A and the C block share their left and right edges, the C heatmap
and D share top and bottom, and the B chord is scaled to the largest size its labels allow.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.patches import PathPatch, Polygon, Wedge
from matplotlib.path import Path
from scipy.cluster import hierarchy
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from common import FIG, GROUP_COL, GROUP_LABEL, INK, MUTED, apply_style, save

D = FIG / "Fig4"
pcoa = pd.read_csv(D / "fig4_pcoa_species.csv", encoding="utf-8-sig")
genus = pd.read_csv(D / "fig4_genus_per_sample.csv", index_col=0, encoding="utf-8-sig")
st = json.load(open(D / "fig4_stats.json", encoding="utf-8"))
alpha = pd.read_csv(FIG.parent / "宏基因组" / "平台导出" / "多样性指数结果_nr_Origin.csv").set_index("Sample")
apply_style()
ORDER = ["A0", "AY", "H0", "M0", "MY"]
samples = pcoa.Sample.tolist()
grp = pcoa.set_index("Sample").Group
genus = genus[samples]
means = pd.DataFrame({g: genus[grp[grp == g].index].mean(axis=1) for g in ORDER})
# genus colours: Clostridium, Lactobacillus, Sporanaerobacter and Garciella keep their Fig 2 colours
GCOL = {"Clostridium": "#eb6834", "Thermoanaerobacterium": "#2a78d6", "Schnuerera": "#e34948",
        "Neomoorella": "#008300", "Symbiobacterium": "#86b6ef", "Lactobacillus": "#eda100",
        "Caldifermentibacillus": "#e87ba4", "Caldibacillus": "#8c564b", "Sporanaerobacter": "#1baf7a",
        "Garciella": "#6250d6", "Heyndrickxia": "#17becf", "Others": "#c3c2b7"}
SLABEL = {s: f"{GROUP_LABEL[s[:2]]}-{s[-1]}" for s in samples}

# ------------------------------------------------------------------ layout grid (inches)
W, H = 7.2, 7.75
XL, XC = 0.6, 3.3        # left column: A frame = C block (row tree + heatmap)
XH = XL + 0.3            # heatmap left edge
XR, XE = 4.62, 7.1       # right column: D frame
HB0 = 0.62
HB1 = HB0 + (XE - XR)    # heatmap body = D frame (square)
TB0, TB1 = 4.6, 7.45     # top row: A frame; B is centred on the same band
fig = plt.figure(figsize=(W, H))


def box(x0, y0, x1, y1):
    """Axes from a box in inches (origin bottom-left)."""
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


# ------------------------------------------------------------------ A  PCoA
ax = box(XL, TB0, XC, TB1)
for g in ORDER:
    p = pcoa[pcoa.Group == g][["PC1", "PC2"]].values
    c = p.mean(axis=0)
    if len(p) >= 3:
        hull = p[ConvexHull(p).vertices]
        ax.add_patch(Polygon(hull, closed=True, facecolor=GROUP_COL[g], alpha=0.16, edgecolor=GROUP_COL[g], lw=0.7))
    else:
        ax.plot(p[:, 0], p[:, 1], color=GROUP_COL[g], lw=3.5, alpha=0.25, solid_capstyle="round")
    for q in p:
        ax.plot([c[0], q[0]], [c[1], q[1]], color=GROUP_COL[g], lw=0.7, zorder=2)
    ax.scatter(p[:, 0], p[:, 1], s=28, color=GROUP_COL[g], edgecolor=INK, linewidth=0.5, zorder=3,
               marker="s" if g.endswith("Y") else "o", label=GROUP_LABEL[g])
    ax.scatter(*c, s=55, marker="+", color=INK, linewidth=0.9, zorder=4)
ax.axhline(0, color=MUTED, lw=0.6, ls=":"); ax.axvline(0, color=MUTED, lw=0.6, ls=":")
ax.set_xlabel(f"PC1 ({st['explained']['PC1'] * 100:.1f}%)"); ax.set_ylabel(f"PC2 ({st['explained']['PC2'] * 100:.1f}%)")
ax.set_xlim(-0.42, 0.8); ax.set_ylim(-0.58, 0.72)
ax.legend(frameon=False, loc="lower right", fontsize=6.4, handletextpad=0.1, labelspacing=0.3, borderaxespad=0.3)
ins = ax.inset_axes([0.6, 0.62, 0.3, 0.3])          # PERMANOVA partition of the 2 x 2 subset
pm = st["permanova_2x2"]
parts = [("NaCl", pm["NaCl"]["R2"], "#eb6834"), ("Ni", pm["Ni"]["R2"], "#eda100"),
         ("NaCl × Ni", pm["NaCl x Ni"]["R2"], "#6250d6")]
parts.append(("Residual", 1 - sum(p[1] for p in parts), "#c3c2b7"))
ins.pie([p[1] for p in parts], colors=[p[2] for p in parts], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=0.8))
for (name, r2, _), wdg in zip(parts, ins.patches):
    ang = np.deg2rad((wdg.theta1 + wdg.theta2) / 2)
    ins.text(1.62 * np.cos(ang), 1.5 * np.sin(ang), f"{name}\n{r2 * 100:.0f}%", ha="center", va="center", fontsize=5.6)
ins.text(0, 0, "R$^2$", ha="center", va="center", fontsize=6.5)
ins.set_aspect("equal")

# ------------------------------------------------------------------ B  chord diagram
BX0, BX1, BY0, BY1 = XC + 0.12, W - 0.03, TB0 - 0.2, TB1 + 0.15   # space the ring and labels may use
ax_B = box(BX0, BY0, BX1, BY1)
gen_order = ["Schnuerera", "Clostridium", "Thermoanaerobacterium", "Neomoorella", "Symbiobacterium",
             "Sporanaerobacter", "Lactobacillus", "Caldifermentibacillus", "Caldibacillus"]
M = means.loc[gen_order].copy()
M.loc["Others"] = 100 - M.sum()
gen_order = gen_order + ["Others"]
GAP, BIG = 1.6, 10.0
tot = M.sum(axis=1)
# genera on the left (anticlockwise from the top), treatments on the right (clockwise from the top)
k_arc, a = {}, 90 + BIG / 2
gspan = 180 - BIG - GAP * (len(gen_order) - 1)
for k in gen_order:
    w = gspan * tot[k] / tot.sum()
    k_arc[k] = [a, a + w]; a += w + GAP
g_arc, a = {}, 90 - BIG / 2
span = (180 - BIG - GAP * (len(ORDER) - 1)) / len(ORDER)   # every treatment sums to 100%, so equal arcs
for g in ORDER:
    g_arc[g] = [a - span, a]; a -= span + GAP


def pt(deg, r=1.0):
    return np.array([r * np.cos(np.deg2rad(deg)), r * np.sin(np.deg2rad(deg))])


def arc(d1, d2, n=24):
    return [pt(d) for d in np.linspace(d1, d2, n)]


g_cursor = {g: g_arc[g][1] for g in ORDER}        # fill each treatment from its top end downwards
k_cursor = {k: k_arc[k][0] for k in gen_order}    # fill each genus from its top end downwards
for k in gen_order:
    for g in ORDER:
        v = M.loc[k, g]
        if v <= 0.2:
            continue
        a2 = g_cursor[g]; a1 = a2 - (g_arc[g][1] - g_arc[g][0]) * v / 100; g_cursor[g] = a1
        b1 = k_cursor[k]; b2 = b1 + (k_arc[k][1] - k_arc[k][0]) * v / tot[k]; k_cursor[k] = b2
        verts = arc(a1, a2) + [np.zeros(2), pt(b1)] + arc(b1, b2)[1:] + [np.zeros(2), pt(a1)]
        codes = [Path.MOVETO] + [Path.LINETO] * 23 + [Path.CURVE3] * 2 + [Path.LINETO] * 23 + [Path.CURVE3] * 2
        ax_B.add_patch(PathPatch(Path(verts, codes), facecolor=GCOL[k], edgecolor="none",
                                 alpha=0.32 if k == "Others" else 0.62, zorder=1))
for name, (d1, d2) in list(k_arc.items()) + list(g_arc.items()):
    is_g = name in g_arc
    ax_B.add_patch(Wedge((0, 0), 1.09, d1, d2, width=0.08, facecolor=GROUP_COL[name] if is_g else GCOL[name],
                         edgecolor=INK, linewidth=0.4, zorder=2))
    mid = (d1 + d2) / 2
    left = 90 < mid % 360 < 270
    ax_B.text(*pt(mid, 1.14), GROUP_LABEL[name] if is_g else name, rotation=(mid + 180 if left else mid) % 360,
              rotation_mode="anchor", ha="right" if left else "left", va="center", fontsize=6.8 if is_g else 6.3,
              style="normal" if (is_g or name == "Others") else "italic")
ax_B.axis("off")


def chord_scale(U, cx):
    """Place the ring centre at cx (inches) with U inches per data unit, ring centred on the A row."""
    cy = (TB0 + TB1) / 2
    ax_B.set_xlim(-(cx - BX0) / U, (BX1 - cx) / U); ax_B.set_ylim(-(cy - BY0) / U, (BY1 - cy) / U)


# largest scale at which every label stays inside the B space; ring pushed to the right edge
renderer = fig.canvas.get_renderer()
for U in np.arange(1.2, 0.5, -0.01):
    chord_scale(U, (BX0 + BX1) / 2)
    ext = [t.get_window_extent(renderer) for t in ax_B.texts]
    lo, hi = min(e.x0 for e in ext) / fig.dpi, max(e.x1 for e in ext) / fig.dpi
    bot, top = min(e.y0 for e in ext) / fig.dpi, max(e.y1 for e in ext) / fig.dpi
    if hi - lo <= BX1 - BX0 and bot >= BY0 and top <= BY1:
        chord_scale(U, (BX0 + BX1) / 2 + (BX1 - hi))
        break

# ------------------------------------------------------------------ C  clustered heatmap
rows = means.max(axis=1).sort_values(ascending=False)
rows = [r for r in rows.index if not r.startswith("unclassified")][:20]
z = np.log10(genus.loc[rows] + 0.01)
z = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1), axis=0)
sp_link = hierarchy.linkage(pdist(genus.T.values, "braycurtis"), "average")   # UPGMA, same distance as PCoA
row_link = hierarchy.linkage(pdist(z.values, "correlation"), "average")
ax_d = box(XH, HB1 + 0.44, XC, HB1 + 0.78)
dn = hierarchy.dendrogram(sp_link, ax=ax_d, color_threshold=0, above_threshold_color=INK, no_labels=True)
ax_d.axis("off")
col_order = [z.columns[i] for i in dn["leaves"]]
ax_r = box(XL, HB0, XH - 0.02, HB1)
dr = hierarchy.dendrogram(row_link, ax=ax_r, orientation="left", color_threshold=0, above_threshold_color=INK,
                          no_labels=True)
ax_r.axis("off"); ax_r.invert_yaxis()
row_order = [z.index[i] for i in dr["leaves"]]
ax_s = box(XH, HB1 + 0.12, XC, HB1 + 0.41)
ax_s.bar(range(len(col_order)), alpha.loc[col_order, "shannon"].values, width=0.75,
         color=[GROUP_COL[s[:2]] for s in col_order], edgecolor=INK, lw=0.3)
ax_s.set_xlim(-0.5, len(col_order) - 0.5); ax_s.set_ylim(0, 5); ax_s.set_yticks([0, 5])
ax_s.set_xticks([]); ax_s.tick_params(labelsize=5.5, length=2)
ax_s.yaxis.tick_right(); ax_s.yaxis.set_label_position("right")
ax_s.set_ylabel("Shannon", fontsize=6.2, rotation=0, ha="left", va="center", labelpad=4)
ax_g = box(XH, HB1 + 0.03, XC, HB1 + 0.09)
ax_g.imshow([[to_rgb(GROUP_COL[s[:2]]) for s in col_order]], aspect="auto")
ax_g.set_xticks([]); ax_g.set_yticks([])
ax_h = box(XH, HB0, XC, HB1)
hm = ax_h.imshow(z.loc[row_order, col_order].values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -2.5, 2.5))
ax_h.set_xticks(range(len(col_order)), [SLABEL[s] for s in col_order], rotation=90, fontsize=5.8)
ax_h.set_yticks(range(len(row_order)), [r.split("_f__")[0] for r in row_order], fontsize=6, style="italic")
ax_h.yaxis.tick_right(); ax_h.tick_params(length=1.5)
cax = box(XL + 0.06, HB1 + 0.12, XL + 0.13, HB1 + 0.62)
cb = fig.colorbar(hm, cax=cax, ticks=[-2, 0, 2]); cb.ax.tick_params(labelsize=5.2, length=1.5); cb.outline.set_linewidth(0.5)
cax.set_title("Z-score", fontsize=6, pad=2)

# ------------------------------------------------------------------ D  salt response with and without Ni
ax = box(XR, HB0, XE, HB1)
EPS = 0.1
cand = means[["A0", "AY", "M0", "MY"]].max(axis=1)
cand = [c for c in cand[cand >= 1].index if not c.startswith("unclassified")]
x = np.log2((means.loc[cand, "AY"] + EPS) / (means.loc[cand, "A0"] + EPS))
y = np.log2((means.loc[cand, "MY"] + EPS) / (means.loc[cand, "M0"] + EPS))
size = means.loc[cand, ["A0", "AY", "M0", "MY"]].max(axis=1)
lim = 8.5
ax.fill_between([-lim, lim], [-lim, lim], lim, color="#e87ba4", alpha=0.07, lw=0)
ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
ax.axhline(0, color=MUTED, lw=0.6, ls=":"); ax.axvline(0, color=MUTED, lw=0.6, ls=":")
for c in cand:
    ax.scatter(x[c], y[c], s=8 + size[c] * 3.2, color=GCOL.get(c, "#c3c2b7"), edgecolor=INK, linewidth=0.45,
               alpha=0.9, zorder=3 if c in GCOL else 2)
LAB = {"Clostridium": (0.3, -1.6), "Thermoanaerobacterium": (-2.4, -1.6), "Lactobacillus": (0.4, 1.1),
       "Caldifermentibacillus": (-1.0, 1.3), "Caldibacillus": (0.8, -1.3), "Schnuerera": (0.3, 1.0),
       "Symbiobacterium": (-1.6, -1.3), "Garciella": (0.5, 0.9), "Heyndrickxia": (-1.0, 1.2),
       "Sporanaerobacter": (0.6, 1.1)}
for c, (dx, dy) in LAB.items():
    if c in cand:
        ax.annotate(c, (x[c], y[c]), (x[c] + dx, y[c] + dy), fontsize=6.2, style="italic",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
ax.text(-lim + 0.4, lim - 0.6, "Favoured by Ni\nunder NaCl", fontsize=6.3, va="top", color="#a8456f")
ax.text(lim - 0.4, -lim + 0.4, "Favoured without Ni\nunder NaCl", fontsize=6.3, va="bottom", ha="right", color=MUTED)
ax.set_xticks([-8, -4, 0, 4, 8]); ax.set_yticks([-8, -4, 0, 4, 8])
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("log$_2$(NaCl / Control)"); ax.set_ylabel("log$_2$(Ni + NaCl / Ni)")
for s in [1, 10, 50]:
    ax.scatter([], [], s=8 + s * 3.2, color="white", edgecolor=INK, linewidth=0.45, label=f"{s}%")
ax.legend(title="Maximum mean abundance", title_fontsize=6.3, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=6.3, handletextpad=0.5, columnspacing=1.8, borderaxespad=0)

# ------------------------------------------------------------------ panel letters: A/C and B/D share columns, rows share a baseline
for letter, xx, yy in [("A", 0.06, TB1), ("B", XR - 0.62, TB1), ("C", 0.06, HB1 + 0.78), ("D", XR - 0.62, HB1 + 0.78)]:
    fig.text(xx / W, (yy + 0.06) / H, letter, fontsize=12, fontweight="bold", va="bottom")
save(fig, D / "Fig4", tif=True)
print(f"Fig4 saved (chord scale {U:.2f} in/unit)")
