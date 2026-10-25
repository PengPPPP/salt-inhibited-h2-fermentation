"""Fig 6: integration of community, functional genes and products, and a conceptual model. Run 13 first.

A  db-RDA of the species community constrained by the liquid products (VIF < 10); H2 group means fitted passively
B  Spearman correlations of key genera with products and key genes (circle size and colour = rho)
C  conceptual model: NaCl without Ni, control and NaCl with Ni (community, osmoadaptation genes, products and H2)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Wedge
from scipy.spatial import ConvexHull

from common import FIG, GROUP_COL, GROUP_LABEL, INK, MUTED, ROOT, VFA_COL, apply_style, save

D = FIG / "Fig6"
st = json.load(open(D / "fig6_stats.json", encoding="utf-8"))
sites = pd.read_csv(D / "fig6_dbrda_sites.csv", encoding="utf-8-sig")
bip = pd.read_csv(D / "fig6_dbrda_biplot.csv", index_col=0, encoding="utf-8-sig")
rho = pd.read_csv(D / "fig6_spearman_rho.csv", index_col=0, encoding="utf-8-sig")
qv = pd.read_csv(D / "fig6_spearman_q.csv", index_col=0, encoding="utf-8-sig")
genus = pd.read_csv(FIG / "Fig4" / "fig4_genus_per_sample.csv", index_col=0, encoding="utf-8-sig")
f4 = json.load(open(FIG / "Fig4" / "fig4_stats.json", encoding="utf-8"))
f5 = json.load(open(FIG / "Fig5" / "fig5_gene_stats.json", encoding="utf-8"))
env = pd.read_csv(ROOT / "env_table_jieyizhi_noA0_1.txt", sep="\t", index_col=0)
apply_style()
ORDER = ["A0", "AY", "H0", "M0", "MY"]
GCOL = {"Clostridium": "#eb6834", "Thermoanaerobacterium": "#2a78d6", "Schnuerera": "#e34948",
        "Neomoorella": "#008300", "Symbiobacterium": "#86b6ef", "Lactobacillus": "#eda100",
        "Caldifermentibacillus": "#e87ba4", "Caldibacillus": "#8c564b", "Sporanaerobacter": "#1baf7a",
        "Garciella": "#6250d6", "Heyndrickxia": "#17becf", "Others": "#c3c2b7"}

W, H = 7.2, 6.2
fig = plt.figure(figsize=(W, H))


def box(x0, y0, x1, y1):
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


TOP0, TOP1 = 3.6, 5.95            # top row: A frame and B matrix share top and bottom

# ------------------------------------------------------------------ A  db-RDA
ax = box(0.55, TOP0, 2.95, TOP1)
for g in ORDER:
    p = sites[sites.Group == g][["dbRDA1", "dbRDA2"]].values
    c = p.mean(axis=0)
    if len(p) >= 3:
        ax.add_patch(Polygon(p[ConvexHull(p).vertices], closed=True, facecolor=GROUP_COL[g], alpha=0.16,
                             edgecolor=GROUP_COL[g], lw=0.7))
    else:
        ax.plot(p[:, 0], p[:, 1], color=GROUP_COL[g], lw=3.5, alpha=0.25, solid_capstyle="round")
    ax.scatter(p[:, 0], p[:, 1], s=26, color=GROUP_COL[g], edgecolor=INK, linewidth=0.5, zorder=3,
               marker="s" if g.endswith("Y") else "o", label=GROUP_LABEL[g])
span = np.abs(sites[["dbRDA1", "dbRDA2"]].values).max()
k = 0.85 * span
NAME = {"Carbohydrate": "Carbohydrates"}
LABEL_POS = {"Propionate": (0.0, 0.03, "left", "bottom"), "Carbohydrate": (0, -0.03, "center", "top"),
             "Caproate": (0.0, -0.035, "left", "top")}
for v, (x, y) in bip.iterrows():
    sig = st["margin"][v]["P"] < 0.05
    ax.annotate("", (x * k, y * k), (0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.0 if sig else 0.7,
                                                            color=INK if sig else MUTED, mutation_scale=7))
    dx, dy, ha, va = LABEL_POS.get(v, (0.02 if x > 0 else -0.02, 0, "left" if x > 0 else "right", "center"))
    ax.text(x * k + dx, y * k + dy, NAME.get(v, v), fontsize=6.2, ha=ha, va=va, color=INK if sig else MUTED)
hx, hy = st["H2_passive"]["direction"]
ax.annotate("", (hx * k, hy * k), (0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.0, ls="--", color="#2a78d6",
                                                          mutation_scale=7))
ax.text(hx * k - 0.02, hy * k + 0.02, "H$_2$ (group means)", fontsize=6.2, ha="right", va="bottom", color="#2a78d6")
ax.axhline(0, color=MUTED, lw=0.6, ls=":"); ax.axvline(0, color=MUTED, lw=0.6, ls=":")
r1, r2 = st["axis_R2"]
ax.set_xlabel(f"dbRDA1 ({r1 * 100:.1f}%)"); ax.set_ylabel(f"dbRDA2 ({r2 * 100:.1f}%)")
lim = span * 1.18
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.text(0.03, 0.03, f"Adj. R$^2$ = {st['R2_adj']:.2f}\nP < 0.001", transform=ax.transAxes, fontsize=6.2,
        va="bottom")
ax.legend(frameon=False, loc="upper left", fontsize=6.0, handletextpad=0.1, labelspacing=0.25, borderaxespad=0.3)

# ------------------------------------------------------------------ B  Spearman matrix
GENERA = ["Clostridium", "Schnuerera", "Symbiobacterium", "Neomoorella", "Thermoanaerobacterium", "Heyndrickxia",
          "Lactobacillus", "Caldifermentibacillus", "Caldibacillus", "Garciella"]
PROD = ["Butyrate", "Acetate", "Ethanol", "Propionate", "Carbohydrate", "Protein"]
GENES = ["ptb", "buk", "hndD", "etfAB", "atoDA", "ldhA", "nhaC", "mrpA-G", "proVWX", "cbiMNOQ"]
COLS = PROD + GENES
CELL = 0.162
BX1 = 7.08
BX0 = BX1 - CELL * (len(COLS) + 0.6)            # small gap between products and genes
BY1 = TOP1
BY0 = BY1 - CELL * len(GENERA)
axb = box(BX0, BY0, BX1, BY1)
xpos = {c: i + (0.6 if i >= len(PROD) else 0) for i, c in enumerate(COLS)}
RN = TwoSlopeNorm(0, -1, 1)
for i, g in enumerate(GENERA):
    for c in COLS:
        r, q = rho.loc[g, c], qv.loc[g, c]
        axb.add_patch(Circle((xpos[c], i), 0.47 * np.sqrt(abs(r)), facecolor=plt.get_cmap("RdBu_r")(RN(r)),
                             edgecolor=INK if q < 0.05 else "none", lw=0.45, alpha=1 if q < 0.05 else 0.45))
        if q < 0.01:
            axb.text(xpos[c], i + 0.02, "**" if q >= 0.001 else "***", fontsize=4.6, ha="center", va="center",
                     color="white" if abs(r) > 0.6 else INK)
for i in range(len(GENERA) + 1):
    axb.plot([-0.5, len(PROD) - 0.5], [i - 0.5] * 2, color="#e4e3dd", lw=0.4)
    axb.plot([len(PROD) + 0.1, len(COLS) + 0.1], [i - 0.5] * 2, color="#e4e3dd", lw=0.4)
for c in COLS:
    axb.plot([xpos[c] - 0.5] * 2, [-0.5, len(GENERA) - 0.5], color="#e4e3dd", lw=0.4)
    axb.plot([xpos[c] + 0.5] * 2, [-0.5, len(GENERA) - 0.5], color="#e4e3dd", lw=0.4)
axb.set_xlim(-0.5, len(COLS) + 0.1); axb.set_ylim(len(GENERA) - 0.5, -0.5); axb.set_aspect("equal")
axb.set_yticks(range(len(GENERA)), GENERA, style="italic", fontsize=6.2)
axb.set_xticks([xpos[c] for c in COLS], [NAME.get(c, c) if c in PROD else c.replace("-", "–") for c in COLS],
               rotation=90, fontsize=6.2)
for lab, c in zip(axb.get_xticklabels(), COLS):
    lab.set_style("normal" if c in PROD else "italic")
axb.tick_params(length=0, pad=2)
for s in axb.spines.values():
    s.set_visible(False)
for i, g in enumerate(GENERA):
    axb.add_patch(Circle((-1.0, i), 0.22, color=GCOL[g], clip_on=False))
axb.tick_params(axis="y", pad=9)
for x0, x1, lab in [(-0.4, len(PROD) - 0.6, "Products"), (len(PROD) + 0.2, len(COLS) + 0.0, "Functional genes")]:
    axb.plot([x0, x1], [-0.85, -0.85], color=INK, lw=0.8, clip_on=False)
    axb.text((x0 + x1) / 2, -1.05, lab, fontsize=6.4, ha="center", va="bottom")
cbx = box(BX0 - 1.25, TOP0 + 0.05, BX0 - 0.6, TOP0 + 0.11)
cb = fig.colorbar(plt.cm.ScalarMappable(RN, "RdBu_r"), cax=cbx, orientation="horizontal", ticks=[-1, 0, 1])
cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1); cb.outline.set_linewidth(0.5)
cbx.set_title("Spearman ρ", fontsize=5.8, pad=2)

# ------------------------------------------------------------------ C  conceptual model
axc = box(0.05, 0.05, 7.15, 3.0)
axc.set_xlim(0, 7.1); axc.set_ylim(0, 2.95); axc.set_aspect("equal"); axc.axis("off")
gm = genus.drop(columns=["A0_1"]).T.groupby(lambda s: s[:2]).mean().T
tc = pd.read_csv(FIG / "FigS3" / "figS3_relief_timecourse.csv", encoding="utf-8-sig")
tc = tc.groupby(["Group", "Time"])[["Butyrate", "Acetate"]].mean()
delta = pd.DataFrame({g: tc.loc[g].iloc[-1] - tc.loc[g].iloc[0] for g in ["A0", "AY", "MY"]}).T
h2 = json.load(open(FIG / "Fig3" / "fig3_stats.json", encoding="utf-8"))["H2_final_mL"]["mean"]
STATES = [("AY", "NaCl (20 g/L)", 1.1), ("A0", "Control", 3.45), ("MY", "Ni (2 mg/L) + NaCl", 5.8)]
TOPG = ["Clostridium", "Schnuerera", "Thermoanaerobacterium", "Lactobacillus", "Caldifermentibacillus",
        "Caldibacillus", "Garciella", "Heyndrickxia"]
TRANSP = [("mrpA-G", "Mrp", "Na$^+$", "out", "#2a78d6"), ("proVWX", "ProVWX", "GB", "in", "#6250d6"),
          ("cbiMNOQ", "Cbi", "Ni$^{2+}$", "in", "#1baf7a")]
ref = {g: max(f5[g]["mean"][s] for s, _, _ in STATES) for g, *_ in TRANSP}
rng = np.random.default_rng(7)
for grp, title, cx in STATES:
    axc.text(cx, 2.83, title, fontsize=7.4, ha="center", va="center", fontweight="bold")
    axc.add_patch(FancyBboxPatch((cx - 0.55, 2.72), 1.1, 0.03, boxstyle="round,pad=0", color=GROUP_COL[grp], lw=0))
    # community donut: dominant genera, Shannon in the centre
    dcx, dcy, R = cx - 0.62, 2.26, 0.32
    comp = gm.loc[TOPG, grp]
    parts = list(comp.items()) + [("Others", 100 - comp.sum())]
    a = 90
    for name, v in parts:
        axc.add_patch(Wedge((dcx, dcy), R, a - 3.6 * v, a, width=0.13, facecolor=GCOL[name], edgecolor="white", lw=0.5))
        a -= 3.6 * v
    axc.text(dcx, dcy, f"H′\n{f4['Shannon']['mean'][grp]:.2f}", fontsize=5.6, ha="center", va="center",
             linespacing=1.0)
    for j, (name, v) in enumerate(comp.sort_values(ascending=False).head(3).items()):
        axc.add_patch(Circle((cx - 0.12, dcy + 0.2 - j * 0.2), 0.04, color=GCOL[name]))
        axc.text(cx - 0.04, dcy + 0.2 - j * 0.2, f"{name} {v:.0f}%", fontsize=5.6, style="italic", va="center")
    # cell
    x0, x1, y0, y1 = cx - 0.95, cx + 0.8, 0.42, 1.42
    axc.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.18",
                                 facecolor="#fbf8f1", edgecolor="#b39b6e", lw=2.2, zorder=1))
    if grp != "A0":                                      # Na+ outside the cell
        for _ in range(16):
            px, py = rng.uniform(x0 - 0.1, x1 + 0.1), rng.uniform(1.8, 1.9)
            axc.add_patch(Circle((px, py), 0.022, color="#2a78d6", alpha=0.45, lw=0))
    if grp == "MY":                                      # added Ni
        for _ in range(6):
            px, py = rng.uniform(x0, x1), rng.uniform(1.8, 1.9)
            axc.add_patch(Circle((px, py), 0.03, color="#1baf7a", lw=0))
    # membrane transporters, opacity scaled to gene abundance; absent = dashed outline
    for k_, (gname, label, ion, way, col) in enumerate(TRANSP):
        tx = x0 + 0.38 + k_ * 0.55
        rel = f5[gname]["mean"][grp] / ref[gname]
        present = f5[gname]["mean"][grp] >= 5
        axc.add_patch(FancyBboxPatch((tx - 0.13, y1 - 0.09), 0.26, 0.18, boxstyle="round,pad=0,rounding_size=0.05",
                                     facecolor=col if present else "white", alpha=0.25 + 0.75 * rel if present else 1,
                                     edgecolor=col, lw=0.8, ls="-" if present else (0, (2, 1.5)), zorder=3))
        axc.text(tx, y1 - 0.2, label, fontsize=5.3, ha="center", va="top", color=INK if present else MUTED)
        if present:
            ys, ye = (y1 + 0.1, y1 + 0.32) if way == "out" else (y1 + 0.32, y1 + 0.1)
            axc.add_patch(FancyArrowPatch((tx + 0.17, ys), (tx + 0.17, ye), arrowstyle="-|>", mutation_scale=5,
                                          lw=0.7, color=col, zorder=3))
            axc.text(tx + 0.24, y1 + 0.22, ion, fontsize=5.2, va="center", color=col)
        else:
            axc.text(tx, y1, "×", fontsize=7, ha="center", va="center", color=MUTED, zorder=4)
    # carbon flow inside the cell, first to last liquid sampling (Fig S3): butyrate formed from sugars and acetate
    # taken up by CoA transfer (AtoDA); band widths follow the net changes (mg COD/L)
    d_but, d_ace = delta.loc[grp, "Butyrate"], delta.loc[grp, "Acetate"]
    sx, sy, bx = x0 + 0.27, 1.0, x0 + 0.95
    axc.text(sx, sy, "Sugars", fontsize=6, ha="center", va="center")
    tw = 1 + 13 * d_but / 2300
    axc.add_patch(FancyArrowPatch((sx + 0.25, sy), (bx - 0.05, sy), mutation_scale=1, color=VFA_COL["Butyrate"],
                                  alpha=0.85, lw=0, shrinkA=0, shrinkB=0,
                                  arrowstyle=f"simple,head_length=6,head_width={tw + 4:.1f},tail_width={tw:.1f}"))
    axc.text(bx, sy, f"Butyrate +{d_but:.0f}", fontsize=5.8, va="center")
    axc.text(sx, 0.62, "Acetate", fontsize=6, ha="center", va="center")
    if d_ace < 0:
        tw = 1 + 13 * -d_ace / 2300
        axc.add_patch(FancyArrowPatch((sx + 0.27, 0.66), (bx - 0.12, sy - 0.06), mutation_scale=1,
                                      color=VFA_COL["Acetate"], alpha=0.85, lw=0, shrinkA=0, shrinkB=0,
                                      arrowstyle=f"simple,head_length=5,head_width={tw + 3:.1f},tail_width={tw:.1f}"))
        axc.text(bx - 0.25, 0.62, f"AtoDA, uptake {-d_ace:.0f}", fontsize=5.4, va="center")
    else:
        axc.plot([sx + 0.27, bx - 0.12], [0.66, sy - 0.06], color=MUTED, lw=0.8, ls=(0, (2, 1.5)))
        axc.text((sx + bx) / 2 - 0.02, 0.81, "×", fontsize=8, ha="center", va="center", color=MUTED)
        axc.text(bx - 0.25, 0.62, f"no uptake (+{d_ace:.0f})", fontsize=5.4, va="center", color=MUTED)
    # H2 leaving the cell, bubble area proportional to the final yield
    hr = 0.07 + 0.16 * np.sqrt(h2[grp] / 43)
    axc.add_patch(FancyArrowPatch((x1 - 0.18, 1.12), (x1 + 0.02, 1.12), arrowstyle="-|>", mutation_scale=5, lw=0.8,
                                  color="#2a78d6", zorder=3))
    axc.add_patch(Circle((x1 + 0.04 + hr, 1.12), hr, facecolor="#2a78d6", alpha=0.85, lw=0, zorder=3))
    axc.text(x1 + 0.04 + hr, 1.12, "H$_2$", fontsize=5.8, ha="center", va="center", color="white", zorder=4)
    axc.text(x1 + 0.04 + hr, 1.12 - hr - 0.06, f"{h2[grp]:.1f} mL", fontsize=5.4, ha="center", va="top")
    # key gene changes relative to the control
    note = {"AY": "$ptb$–$buk$ ↓98%, $etfAB$ ↓82%\n$atoDA$ not detected, $ech$ ↑3.6×",
            "A0": "$ptb$–$buk$ and $hndD$ abundant\nbutyrate-type fermentation",
            "MY": "$etfAB$ ↑3.2×, $atoDA$ ↑1.8× (vs Ni)\n$ptb$–$buk$ ↓80%"}[grp]
    axc.text(cx - 0.075, 0.2, note, fontsize=5.4, ha="center", va="center", linespacing=1.1, color=INK)
# arrows from the control towards both salt states
for xa, xb, lab in [(2.9, 1.7, "+ NaCl"), (4.0, 4.95, "+ NaCl + Ni")]:
    axc.add_patch(FancyArrowPatch((xa, 2.83), (xb, 2.83), arrowstyle="-|>", mutation_scale=8, lw=1.0, color=INK))
    axc.text((xa + xb) / 2, 2.88, lab, fontsize=5.8, ha="center", va="bottom")

# ------------------------------------------------------------------ panel letters
for letter, xx, yy in [("A", 0.06, TOP1), ("B", BX0 - 1.45, TOP1), ("C", 0.06, 2.92)]:
    fig.text(xx / W, (yy + 0.04) / H, letter, fontsize=12, fontweight="bold", va="bottom")
save(fig, D / "Fig6", tif=True)
print("Fig6 saved")
