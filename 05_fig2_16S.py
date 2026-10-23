"""Fig 2: 16S community of the salt-gradient batch. Run 01 first (uses its H2 / butyrate group means).

Inputs: 16S/平台导出/Alpha多样性指数表_*.csv (platform alpha tables), 16S/16S_属水平组均值.csv (genus group means),
        图表/Fig2/Fig2A_PCoA坐标.csv (Bray-Curtis PCoA sample scores)
Outputs: 16S/16S_alpha_Y0-Y4.csv, 图表/Fig2/Fig2.(png|pdf|tif), 图表/Fig2/fig2_stats.json
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy import stats

from common import FIG, INK, NACL, RAMP, S16, SALT_GROUPS, MUTED, apply_style, pfmt, quadratic_test, save, \
    spearman_exact, tag

OUT = FIG / "Fig2"
OUT.mkdir(parents=True, exist_ok=True)

# alpha diversity: merge the two platform exports, keep the salt-gradient samples
tables = [pd.read_csv(p, encoding="utf-8-sig") for p in sorted((S16 / "平台导出").glob("Alpha多样性指数表_*.csv"))]
for t in tables:
    t.rename(columns={t.columns[0]: "Sample"}, inplace=True)
alpha = tables[0]
for t in tables[1:]:
    alpha = alpha.merge(t, on="Sample")
alpha = alpha[alpha.Sample.str.startswith("Y")].copy()
alpha["Group"] = alpha.Sample.str[:2]
alpha["NaCl_gL"] = alpha.Group.map(NACL)
alpha = alpha.sort_values("Sample")
alpha.to_csv(S16 / "16S_alpha_Y0-Y4.csv", index=False, encoding="utf-8-sig")

genus = pd.read_csv(S16 / "16S_属水平组均值.csv", index_col=0, encoding="utf-8-sig")[SALT_GROUPS]
pcoa = pd.read_csv(OUT / "Fig2A_PCoA坐标.csv")
f1 = json.load(open(FIG / "Fig1" / "fig1_stats.json", encoding="utf-8"))
h2 = pd.Series(f1["H2_67h"]["mean"])[SALT_GROUPS]
but = pd.Series(f1["Butyrate_pct_67h"]["mean"])[SALT_GROUPS]

# statistics
x = alpha.NaCl_gL.values.astype(float)
res = {}
for m in ["Sobs", "Shannon", "Simpson", "Simpsoneven", "Pielou_e", "Pd"]:
    b, pq = quadratic_test(x, alpha[m].values)
    rho, prho = stats.spearmanr(x, alpha[m])
    res[m] = {"mean": alpha.groupby("Group")[m].mean().round(3).to_dict(),
              "sd": alpha.groupby("Group")[m].std().round(3).to_dict(),
              "p_anova": stats.f_oneway(*[alpha[alpha.Group == g][m] for g in SALT_GROUPS]).pvalue,
              "p_quad": pq, "rho": rho, "p_rho": prho}
for gname in ["Clostridium", "Lactobacillus", "Sporanaerobacter"]:
    res[gname] = {k: dict(zip(["rho", "p_exact"], spearman_exact(genus.loc[gname].values, v.values)))
                  for k, v in [("H2", h2), ("Butyrate_pct", but)]}
json.dump(res, open(OUT / "fig2_stats.json", "w", encoding="utf-8"), indent=1, default=float)

# figure
apply_style()
ramp_handles = [Line2D([], [], color=RAMP[g], marker="o", ms=4, lw=0, label=str(NACL[g])) for g in SALT_GROUPS]
fig = plt.figure(figsize=(7.1, 4.9))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.45, left=0.075, right=0.985, top=0.95, bottom=0.17)

ax = fig.add_subplot(gs[0, 0]); tag(ax, "A")
label_pos = {"Y0": (-0.29, 0.07), "Y1": (0.10, -0.31), "Y2": (0.07, -0.06), "Y3": (-0.07, 0.115), "Y4": (0.27, 0.05)}
for g in SALT_GROUPS:
    p = pcoa[pcoa.group == g][["PC1", "PC2"]].values
    ax.add_patch(Polygon(p, closed=True, facecolor=RAMP[g], alpha=0.15, edgecolor=RAMP[g], lw=0.6, zorder=1))
    ax.scatter(p[:, 0], p[:, 1], s=22, color=RAMP[g], edgecolor=INK, linewidth=0.4, zorder=3)
    ax.text(*label_pos[g], f"{NACL[g]} g/L", ha="center", va="center", fontsize=6.5, zorder=4)
ax.axhline(0, color=MUTED, lw=0.6, ls=":"); ax.axvline(0, color=MUTED, lw=0.6, ls=":")
ax.set_xlabel("PC1 (46.86%)"); ax.set_ylabel("PC2 (22.38%)"); ax.set_xlim(-0.42, 0.42); ax.set_ylim(-0.36, 0.33)
ax.text(0.03, 0.03, "ANOSIM R = 0.69\nP = 0.001", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5)

ax = fig.add_subplot(gs[0, 1:])
ax.text(-0.115, 1.03, "B", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
show = ["Clostridium", "Sporanaerobacter", "Lactobacillus", "Litchfieldia", "Ureibacillus", "Garciella",
        "Tepidimicrobium", "Unclassified Bacillales"]
cols = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#2a78d6"]
comp = genus.copy()
comp.loc["Unclassified Bacillales"] = comp.loc[["unclassified_f__Bacillaceae", "unclassified_o__Bacillales"]].sum()
comp = comp.loc[show]
comp.loc["Others"] = 100 - comp.sum()
bottom = np.zeros(5); xs = np.arange(5)
for name, c in zip(show + ["Others"], cols + ["#c3c2b7"]):
    ax.bar(xs, comp.loc[name], bottom=bottom, width=0.6, color=c, edgecolor=INK, linewidth=0.4,
           label=name if name in ("Others", "Unclassified Bacillales") else f"$\\it{{{name}}}$")
    bottom += comp.loc[name].values
ax.set_xticks(xs, [str(NACL[g]) for g in SALT_GROUPS]); ax.set_xlabel("NaCl (g/L)")
ax.set_ylabel("Relative abundance (%)"); ax.set_ylim(0, 100); ax.set_xlim(-0.5, 4.5)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.8, handlelength=1.0,
          labelspacing=0.35, reverse=True)
pos = ax.get_position(); ax.set_position([pos.x0, pos.y0, pos.width * 0.62, pos.height])


def by_nacl(ax, col, ylabel):
    for g in SALT_GROUPS:
        v = alpha[alpha.Group == g][col].values
        ax.scatter(NACL[g] + np.array([-0.6, 0, 0.6]), v, s=9, color=RAMP[g], edgecolor="white", linewidth=0.4, zorder=3)
        ax.errorbar(NACL[g], v.mean(), yerr=v.std(ddof=1), fmt="_", color=INK, ms=7, mew=1.0, elinewidth=0.7,
                    capsize=1.8, zorder=4)
    ax.set_xticks([0, 5, 10, 15, 20]); ax.set_xlim(-2.5, 22.5); ax.set_xlabel("NaCl (g/L)"); ax.set_ylabel(ylabel)


xx = np.linspace(0, 20, 100)
ax = fig.add_subplot(gs[1, 0]); tag(ax, "C")
by_nacl(ax, "Shannon", "Shannon index")
b, pq = quadratic_test(x, alpha.Shannon.values)
ax.plot(xx, b[0] + b[1] * xx + b[2] * xx ** 2, color=INK, lw=0.9, ls="--")
ax.text(0.97, 0.04, f"Quadratic fit, {pfmt(pq)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
ax.set_ylim(1.8, 3.0)

ax = fig.add_subplot(gs[1, 1]); tag(ax, "D")
by_nacl(ax, "Simpsoneven", "Simpson evenness")
r = stats.linregress(x, alpha.Simpsoneven.values)
ax.plot(xx, r.intercept + r.slope * xx, color=INK, lw=0.9, ls="--")
ax.text(0.03, 0.96, f"ρ = {res['Simpsoneven']['rho']:.2f}, {pfmt(res['Simpsoneven']['p_rho'])}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
ax.set_ylim(0.08, 0.23)

ax = fig.add_subplot(gs[1, 2]); tag(ax, "E")
for gname, mk in [("Clostridium", "o"), ("Lactobacillus", "s")]:
    for g, vi, hi in zip(SALT_GROUPS, genus.loc[gname].values, h2.values):
        ax.scatter(vi, hi, s=26, marker=mk, color=RAMP[g], edgecolor=INK, linewidth=0.5, zorder=3)
ax.set_xlabel("Relative abundance (%)"); ax.set_ylabel("H$_2$ at 67 h (mL)"); ax.set_xlim(0, 55); ax.set_ylim(18, 60)
ax.legend(handles=[Line2D([], [], marker="o", lw=0, mfc="white", mec=INK, ms=4.5, label="$\\it{Clostridium}$"),
                   Line2D([], [], marker="s", lw=0, mfc="white", mec=INK, ms=4.5, label="$\\it{Lactobacillus}$")],
          frameon=False, loc="upper center", fontsize=6.3, handlelength=0.8, borderaxespad=0.2)
fig.legend(handles=ramp_handles, title="NaCl (g/L)", title_fontsize=6.5, frameon=False, fontsize=6.5, ncol=5,
           loc="lower center", bbox_to_anchor=(0.5, 0.0), handletextpad=0.2, columnspacing=1.0)
save(fig, OUT / "Fig2", tif=True)
print({m: round(res[m]["p_quad"], 4) for m in ["Sobs", "Shannon", "Simpson"]}, "Lactobacillus vs H2", res["Lactobacillus"]["H2"])
