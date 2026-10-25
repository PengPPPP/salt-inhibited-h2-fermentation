"""Fig 5: functional genes (metagenome) and transcriptional response (metatranscriptome). Run 12 first.

A  fermentation map: each gene carries two tiles, the log2 fold change of its abundance under NaCl without Ni
   (NaCl / Control) and with Ni (Ni + NaCl / Ni); product tiles use measured concentrations
B  bubble matrix of osmoadaptation and Ni-related genes (size = CPM, colour = row z-score of log10 CPM)
C  metatranscriptome: DEG counts per comparison (bars) above KEGG pathway enrichment of up- and down-regulated genes
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from common import FIG, GROUP_COL, GROUP_LABEL, INK, MUTED, ROOT, VFA_COL, apply_style, save

D = FIG / "Fig5"
st = json.load(open(D / "fig5_gene_stats.json", encoding="utf-8"))
f3 = json.load(open(FIG / "Fig3" / "fig3_stats.json", encoding="utf-8"))
env = pd.read_csv(ROOT / "env_table_jieyizhi_noA0_1.txt", sep="\t", index_col=0)
env_mean = env.groupby(env.index.str[:2]).mean()
apply_style()
ORDER = ["A0", "AY", "H0", "M0", "MY"]
CMAP = "RdBu_r"
FC_NORM = TwoSlopeNorm(0, -6, 6)
stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

W, H = 7.2, 6.82
fig = plt.figure(figsize=(W, H))
renderer = fig.canvas.get_renderer()


def box(x0, y0, x1, y1):
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


# ------------------------------------------------------------------ A  fermentation map
AX0, AX1, AY0, AY1 = 0.12, 7.12, 3.54, 6.6
ax = box(AX0, AY0, AX1, AY1)
ax.set_xlim(0, 10.2); ax.set_ylim(0, 4.85); ax.axis("off")
UX, UY = (AX1 - AX0) / 10.2, (AY1 - AY0) / 4.85          # inches per data unit
TW, TH = 0.13 / UX, 0.13 / UY                          # square tiles of 0.13 in

NODES = {"Glucose": (0.9, 4.55), "Pyruvate": (0.9, 2.75), "Lactate": (0.9, 0.55), "Acetyl-CoA": (3.9, 2.75),
         "Ethanol": (2.75, 4.55), "Acetate": (5.0, 4.55), "Acetoacetyl-CoA": (3.9, 0.55),
         "3-Hydroxybutyryl-CoA": (6.2, 0.55), "Crotonyl-CoA": (8.75, 0.55), "Butyryl-CoA": (8.75, 2.75),
         "Butyrate": (8.75, 4.55), "Fd$_{red}$": (5.45, 2.75), "H$_2$": (7.25, 2.75)}
PRODUCT_FILL = {"Lactate": "#d9d8d1", "Ethanol": VFA_COL["Ethanol"], "Acetate": VFA_COL["Acetate"],
                "Butyrate": VFA_COL["Butyrate"], "H$_2$": "#2a78d6"}
extent = {}
for name, (x, y) in NODES.items():
    t = ax.text(x, y, name, ha="center", va="center", fontsize=6.6, zorder=4)
    bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
    w, h = bb.width + 0.24, 0.36
    fc = PRODUCT_FILL.get(name)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=fc + "33" if fc else "#f1f0ea", edgecolor=fc or MUTED, lw=0.8, zorder=3))
    extent[name] = (x - w / 2, x + w / 2, y - h / 2, y + h / 2)


def edge(a, b):
    """Arrow between node boxes a -> b."""
    (xa, ya), (xb, yb) = NODES[a], NODES[b]
    ea, eb = extent[a], extent[b]
    if abs(ya - yb) < 1e-6:
        p0, p1 = ((ea[1], ya), (eb[0], yb)) if xb > xa else ((ea[0], ya), (eb[1], yb))
    elif abs(xa - xb) < 1e-6:
        p0, p1 = ((xa, ea[3]), (xb, eb[2])) if yb > ya else ((xa, ea[2]), (xb, eb[3]))
    else:
        p0 = (xa + (0.25 if xb > xa else -0.25), ea[3] if yb > ya else ea[2])
        p1 = (xb, eb[2] if yb > ya else eb[3])
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=1.0, color=INK,
                                 shrinkA=1.5, shrinkB=1.5, zorder=2))


for a, b in [("Glucose", "Pyruvate"), ("Pyruvate", "Lactate"), ("Pyruvate", "Acetyl-CoA"), ("Acetyl-CoA", "Ethanol"),
             ("Acetyl-CoA", "Acetate"), ("Acetyl-CoA", "Acetoacetyl-CoA"), ("Acetoacetyl-CoA", "3-Hydroxybutyryl-CoA"),
             ("3-Hydroxybutyryl-CoA", "Crotonyl-CoA"), ("Crotonyl-CoA", "Butyryl-CoA"), ("Butyryl-CoA", "Butyrate"),
             ("Fd$_{red}$", "H$_2$")]:
    edge(a, b)


def tiles(x, y, v1, v2, p=None):
    """Two fold-change tiles starting at x (data units), centred on y; returns the right end."""
    for i, v in enumerate((v1, v2)):
        if v is None:
            ax.add_patch(Rectangle((x + i * (TW + 0.02), y - TH / 2), TW, TH, facecolor="white", edgecolor=MUTED,
                                   lw=0.5, hatch="////", zorder=5))
        else:
            ax.add_patch(Rectangle((x + i * (TW + 0.02), y - TH / 2), TW, TH, facecolor=plt.get_cmap(CMAP)(FC_NORM(v)),
                                   edgecolor=INK, lw=0.4, zorder=5))
    end = x + 2 * TW + 0.02
    if p is not None and stars(p):
        ax.text(end + 0.03, y + 0.02, stars(p), fontsize=6, va="center", ha="left", zorder=5)
    return end


def gene(name, x, y, ha="left"):
    """Italic gene name followed by its two tiles; ha='right' ends the block (without stars) at x."""
    s = st[name]
    t = ax.text(0, y, name.replace("-", "–"), fontsize=6.4, style="italic", va="center", ha="left", zorder=5)
    w = t.get_window_extent(renderer).transformed(ax.transData.inverted()).width
    x0 = x if ha == "left" else x - (w + 0.05 + 2 * TW + 0.02)
    t.set_x(x0)
    tiles(x0 + w + 0.05, y, s["log2fc_noNi"], s["log2fc_Ni"], s["p_int_Ni"])


def note(x, y, text, ha="left"):
    ax.text(x, y, text, fontsize=5.9, color=MUTED, ha=ha, va="center", zorder=5)


note(0.75, 3.65, "Glycolysis", ha="right")
gene("ldh", 1.05, 1.9); gene("ldhA", 1.05, 1.58)
note(1.62, 3.72, "PFOR (+ Fd$_{red}$)")
gene("porABCD", 1.62, 3.4); gene("nifJ", 1.62, 3.08); gene("pflB", 1.62, 2.43)
gene("adhE", 2.78, 3.98, ha="right")
gene("pta", 4.55, 3.42); gene("ackA", 4.55, 3.1)
gene("thl", 4.05, 1.65)
gene("hbd", 4.68, 0.86)
gene("crt", 7.4, 0.86)
gene("bcd", 8.9, 1.9); gene("etfAB", 8.9, 1.58); note(8.9, 1.26, "(+ Fd$_{red}$)")
gene("ptb", 8.9, 3.82); gene("buk", 8.9, 3.5); gene("atoDA", 8.25, 3.66, ha="right")
gene("hndD", 5.85, 3.05); gene("echA-F", 5.85, 2.45)

# measured products: log2 fold change of concentration (H2: final cumulative yield of the gas bottles)
fc = lambda c: (float(np.log2(env_mean.loc["AY", c] / env_mean.loc["A0", c])),
                float(np.log2(env_mean.loc["MY", c] / env_mean.loc["M0", c])))
h2 = f3["H2_final_mL"]["mean"]
for name, v in [("Ethanol", fc("Ethanol")), ("Acetate", fc("Acetate")), ("Butyrate", fc("Butyrate")),
                ("H$_2$", (np.log2(h2["AY"] / h2["A0"]), np.log2(h2["MY"] / h2["M0"])))]:
    e = extent[name]
    tiles(e[1] + 0.06, NODES[name][1], *v)

note(NODES["Lactate"][0], extent["Lactate"][2] - 0.17, "not measured", ha="center")

# legend: tile meaning, colour scale, interaction stars
lx, ly = 2.0, 1.05
tiles(lx, ly, -3, 3)
ax.text(lx + TW / 2, ly - 0.2, "−Ni", fontsize=5.6, ha="center", va="top")
ax.text(lx + TW * 1.5 + 0.02, ly - 0.2, "+Ni", fontsize=5.6, ha="center", va="top")
ax.text(lx + 2 * TW + 0.12, ly, "log$_2$ FC under NaCl\n* Ni × NaCl, P < 0.05", fontsize=5.6, va="center")
cax = ax.inset_axes([lx, 0.42, 1.15, 0.09], transform=ax.transData)
cb = fig.colorbar(plt.cm.ScalarMappable(FC_NORM, CMAP), cax=cax, orientation="horizontal", ticks=[-6, -3, 0, 3, 6])
cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1); cb.outline.set_linewidth(0.5)

# ------------------------------------------------------------------ B  osmoadaptation and Ni-related genes
ROWS = [("Na$^+$ extrusion", ["mrpA-G", "nhaC"]), ("Compatible solute\nuptake", ["proVWX", "opuA-C", "opuD"]),
        ("Ni uptake and\n[NiFe] maturation", ["cbiMNOQ", "nikR", "hypA-F"])]
genes = [g for _, gs in ROWS for g in gs]
BY0, BY1 = 0.72, 2.72
BX0, BX1 = 1.55, 1.55 + 0.34 * len(ORDER)
axb = box(BX0, BY0, BX1, BY1)
cpm = pd.DataFrame({g: st[g]["mean"] for g in genes}).T[ORDER]
zl = np.log10(cpm + 1)
zl = zl.sub(zl.mean(axis=1), axis=0).div(zl.std(axis=1), axis=0)
size = lambda v: 6 + 160 * np.sqrt(v / 1500)
yy = {g: i for i, g in enumerate(genes)}
for g in genes:
    for j, grp in enumerate(ORDER):
        axb.scatter(j, yy[g], s=size(cpm.loc[g, grp]), c=[zl.loc[g, grp]], cmap=CMAP, norm=TwoSlopeNorm(0, -1.8, 1.8),
                    edgecolor=INK, linewidth=0.4, zorder=3)
    axb.text(len(ORDER) - 0.35, yy[g], stars(st[g]["p_int_Ni"]), fontsize=6.4, va="center", ha="left")
for k in range(1, len(genes)):
    axb.axhline(k - 0.5, color="#e4e3dd", lw=0.5, zorder=1)
axb.set_xlim(-0.55, len(ORDER) - 0.45); axb.set_ylim(len(genes) - 0.5, -0.5)
axb.set_yticks(range(len(genes)), [g.replace("-", "–") for g in genes], style="italic", fontsize=6.4)
axb.set_xticks(range(len(ORDER)), [GROUP_LABEL[g] for g in ORDER], rotation=40, ha="right", fontsize=6.4)
axb.tick_params(length=1.5)
for j, g in enumerate(ORDER):
    axb.get_xticklabels()[j].set_color(INK)
    axb.add_patch(Rectangle((j - 0.42, len(genes) - 0.62), 0.84, 0.1, color=GROUP_COL[g], clip_on=False, zorder=4))
axb.text(len(ORDER) - 0.32, -0.75, "Ni × NaCl", fontsize=5.6, ha="left", va="bottom", rotation=0)
# category brackets on the left
y = -0.5
for label, gs in ROWS:
    y0, y1 = yy[gs[0]] - 0.38, yy[gs[-1]] + 0.38
    axb.plot([-2.3, -2.3], [y0, y1], color=INK, lw=0.8, clip_on=False)
    axb.text(-2.45, (y0 + y1) / 2, label, fontsize=5.9, ha="right", va="center", linespacing=1.05)
# legends above B: bubble size and colour
xd = lambda inch: (inch - BX0) / 0.34 - 0.55          # inches -> B data x
for v, xp in [(10, xd(0.78)), (100, xd(1.12)), (1000, xd(1.6))]:
    axb.scatter(xp, -1.3, s=size(v), color="white", edgecolor=INK, linewidth=0.4, clip_on=False)
    axb.text(xp + 0.1 + 0.022 * np.sqrt(size(v)), -1.3, f"{v}", fontsize=5.6, va="center")
axb.text(xd(0.36), -1.3, "CPM", fontsize=5.8, va="center", ha="left")
cbx = box(2.3, BY1 + 0.25, 2.9, BY1 + 0.31)
cb = fig.colorbar(plt.cm.ScalarMappable(TwoSlopeNorm(0, -1.8, 1.8), CMAP), cax=cbx, orientation="horizontal", ticks=[-1.5, 0, 1.5])
cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1); cb.outline.set_linewidth(0.5)
cbx.set_title("Z-score", fontsize=5.8, pad=2)

# ------------------------------------------------------------------ C  metatranscriptome
MT = ROOT / "宏转录组"
deg = pd.read_csv(MT / "表达量差异分析" / "差异基因数目统计表_diffexp_Origin.csv", encoding="utf-8-sig").set_index("diff_group")
COMP = [("AY_vs_A0", "NaCl vs\nControl"), ("M0_vs_A0", "Ni vs\nControl"), ("MY_vs_M0", "Ni + NaCl\nvs Ni")]
PATH = [("ko03010", "Ribosome"), ("ko00970", "Aminoacyl-tRNA biosynthesis"), ("ko00190", "Oxidative phosphorylation"),
        ("ko02030", "Bacterial chemotaxis"), ("ko02010", "ABC transporters"), ("ko02060", "Phosphotransferase system"),
        ("ko00010", "Glycolysis / gluconeogenesis"), ("ko00620", "Pyruvate metabolism"),
        ("ko00650", "Butanoate metabolism"), ("ko00640", "Propanoate metabolism")]
cols = [(c, d) for c, _ in COMP for d in ("up", "down")]
enr = {}
for c, d in cols:
    f = next((MT / "目标基因集分析" / "目标基因功能富集分析" / "KEGG功能富集分析" / f"{c}_{d}").glob("*.csv"))
    enr[(c, d)] = pd.read_csv(f, encoding="utf-8-sig").set_index("Pathway id")
CX0 = 4.95
CX1 = CX0 + 0.34 * len(cols)
axc = box(CX0, BY0, CX1, BY1)
PNORM = Normalize(0, 20)
PCMAP = LinearSegmentedColormap.from_list("p", ["#fde6c9", "#eb6834", "#7a2410"])
gsize = lambda n: 4 + 110 * np.sqrt(n / 500)
for j, cd in enumerate(cols):
    for i, (pid, _) in enumerate(PATH):
        r = enr[cd].loc[pid]
        q = r["P_adjust"]
        if q < 0.05:
            axc.scatter(j, i, s=gsize(r["Gene.no"]), c=[min(-np.log10(q), 20)], cmap=PCMAP, norm=PNORM,
                        edgecolor=INK, linewidth=0.4, zorder=3)
        else:
            axc.scatter(j, i, s=gsize(r["Gene.no"]), facecolor="none", edgecolor="#b5b4ad", linewidth=0.5, zorder=3)
for k in range(1, len(PATH)):
    axc.axhline(k - 0.5, color="#e4e3dd", lw=0.5, zorder=1)
for k in (2, 4):
    axc.axvline(k - 0.5, color=MUTED, lw=0.6, zorder=1)
axc.set_xlim(-0.55, len(cols) - 0.45); axc.set_ylim(len(PATH) - 0.5, -0.5)
axc.set_yticks(range(len(PATH)), [p for _, p in PATH], fontsize=6.3)
axc.set_xticks(range(len(cols)), ["Up", "Down"] * 3, fontsize=6.2)
axc.tick_params(length=1.5)
# DEG bars aligned with the columns
axd = box(CX0, BY1 + 0.08, CX1, BY1 + 0.5)
for j, (c, d) in enumerate(cols):
    n = deg.loc[c, "up" if d == "up" else "Down"]
    axd.bar(j, n / 1000, width=0.62, color="#eb6834" if d == "up" else "#2a78d6", edgecolor=INK, lw=0.4)
    axd.text(j, n / 1000 + 0.4, f"{n / 1000:.1f}", fontsize=5.4, ha="center", va="bottom")
axd.set_xlim(-0.55, len(cols) - 0.45); axd.set_ylim(0, 17.5); axd.set_xticks([])
axd.set_yticks([0, 15]); axd.tick_params(labelsize=5.4, length=1.5)
axd.set_ylabel("DEGs\n(× 10$^3$)", fontsize=5.8, rotation=0, ha="right", va="center", labelpad=2)
for side in ("top", "right"):
    axd.spines[side].set_visible(False)
for k, (_, name) in enumerate(COMP):
    axd.text(2 * k + 0.5, 19.0, name, fontsize=6.2, ha="center", va="bottom", linespacing=1.0)
# legends below C
for v, xp in [(50, 0.0), (200, 0.95), (500, 1.95)]:
    axc.scatter(xp, len(PATH) + 1.25, s=gsize(v), color="white", edgecolor=INK, linewidth=0.4, clip_on=False)
    axc.text(xp + 0.1 + 0.022 * np.sqrt(gsize(v)), len(PATH) + 1.25, f"{v}", fontsize=5.6, va="center")
axc.text(-0.4, len(PATH) + 1.25, "Genes", fontsize=5.8, va="center", ha="right")
cbc = box(CX0 + 1.18, BY0 - 0.6, CX1 - 0.05, BY0 - 0.53)
cb = fig.colorbar(plt.cm.ScalarMappable(PNORM, PCMAP), cax=cbc, orientation="horizontal", ticks=[0, 10, 20])
cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1); cb.outline.set_linewidth(0.5)
cbc.set_title("−log$_{10}$ P$_{adj}$", fontsize=5.8, pad=2)

# ------------------------------------------------------------------ panel letters
# B and C sit beside the first row of their panels (B legends, C DEG bars)
for letter, xx, yy_ in [("A", 0.06, AY1 + 0.06), ("B", 0.06, BY1 + 0.3), ("C", CX0 - 1.05, BY1 + 0.3)]:
    fig.text(xx / W, yy_ / H, letter, fontsize=12, fontweight="bold", va="bottom")
save(fig, D / "Fig5", tif=True)
print("Fig5 saved")
