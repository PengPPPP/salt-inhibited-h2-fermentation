"""Fig S4: KEGG pathway enrichment of up- and down-regulated genes (metatranscriptome, platform DEG lists).

For each comparison and direction the ten pathways with the lowest adjusted P are shown; global and overview maps and
the Human Diseases / Organismal Systems categories are left out.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

from common import FIG, INK, MUTED, ROOT, apply_style, save

D = FIG / "FigS4"
D.mkdir(parents=True, exist_ok=True)
ENR = ROOT / "宏转录组" / "目标基因集分析" / "目标基因功能富集分析" / "KEGG功能富集分析"
COMP = [("AY_vs_A0", "NaCl vs Control"), ("M0_vs_A0", "Ni vs Control"), ("MY_vs_M0", "Ni + NaCl vs Ni")]
TOP = 10
PNORM = Normalize(0, 20)
PCMAP = LinearSegmentedColormap.from_list("p", ["#fde6c9", "#eb6834", "#7a2410"])
gsize = lambda n: 4 + 110 * np.sqrt(n / 500)

apply_style()
fig, axs = plt.subplots(3, 2, figsize=(7.1, 8.2))
fig.subplots_adjust(left=0.27, right=0.985, top=0.95, bottom=0.1, wspace=1.25, hspace=0.42)
rows = []
for i, (comp, title) in enumerate(COMP):
    for j, d in enumerate(["up", "down"]):
        t = pd.read_csv(next((ENR / f"{comp}_{d}").glob("*.csv")), encoding="utf-8-sig")
        t = t[(t["Second Category"] != "Global and overview maps")
              & ~t["First Category"].isin(["Human Diseases", "Organismal Systems"])]
        t = t.sort_values("P_adjust").head(TOP).iloc[::-1]
        ax = axs[i, j]
        sig = t.P_adjust < 0.05
        ax.scatter(t["Rich factor"][sig], np.arange(len(t))[sig], s=gsize(t["Gene.no"][sig]),
                   c=np.minimum(-np.log10(t.P_adjust[sig]), 20), cmap=PCMAP, norm=PNORM, edgecolor=INK, lw=0.4, zorder=3)
        ax.scatter(t["Rich factor"][~sig], np.arange(len(t))[~sig], s=gsize(t["Gene.no"][~sig]), facecolor="none",
                   edgecolor="#b5b4ad", lw=0.6, zorder=3)
        ax.set_yticks(range(len(t)), t.Description, fontsize=6.2)
        ax.set_ylim(-0.7, len(t) - 0.3)
        lo, hi = t["Rich factor"].min(), t["Rich factor"].max()
        pad = (hi - lo) * 0.15 + 0.004
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(axis="x", labelsize=6.2)
        ax.set_xlabel("Rich factor", fontsize=7)
        ax.set_title(f"{title}: {'up' if d == 'up' else 'down'}-regulated", fontsize=7.2, fontweight="bold", pad=4)
        ax.grid(axis="y", color="#e4e3dd", lw=0.4, zorder=0)
        rows += [{"Comparison": comp, "Direction": d, **r} for r in t.iloc[::-1][["Pathway id", "Description", "Gene.no",
                                                                                     "Rich factor", "P_adjust"]].to_dict("records")]
# letters: left column at the figure edge, right column just right of the left panels; row top = axes top
for i in range(3):
    top = axs[i, 0].get_position().y1 + 0.012
    fig.text(0.01, top, "ACE"[i], fontsize=12, fontweight="bold", va="bottom")
    fig.text(axs[i, 0].get_position().x1 + 0.03, top, "BDF"[i], fontsize=12, fontweight="bold", va="bottom")
pd.DataFrame(rows).to_csv(D / "figS4_enrichment_top10.csv", index=False, encoding="utf-8-sig")

# shared legends: gene number and adjusted P
lax = fig.add_axes([0.06, 0.012, 0.45, 0.04]); lax.axis("off"); lax.set_xlim(0, 1); lax.set_ylim(0, 1)
lax.text(0.0, 0.5, "Genes", fontsize=6.6, va="center")
for v, x in [(50, 0.2), (200, 0.4), (500, 0.62)]:
    lax.scatter(x, 0.5, s=gsize(v), facecolor="white", edgecolor=INK, lw=0.4, clip_on=False)
    lax.text(x + 0.05, 0.5, str(v), fontsize=6.2, va="center")
lax.scatter(0.82, 0.5, s=gsize(100), facecolor="none", edgecolor="#b5b4ad", lw=0.6)
lax.text(0.86, 0.5, "P$_{adj}$ ≥ 0.05", fontsize=6.2, va="center", color=MUTED)
cax = fig.add_axes([0.66, 0.03, 0.25, 0.012])
cb = fig.colorbar(plt.cm.ScalarMappable(PNORM, PCMAP), cax=cax, orientation="horizontal", ticks=[0, 10, 20])
cb.ax.tick_params(labelsize=5.8, length=1.5, pad=1); cb.outline.set_linewidth(0.5)
cax.set_title("−log$_{10}$ P$_{adj}$", fontsize=6.4, pad=2)
save(fig, D / "FigS4")
print("FigS4 saved")
