"""Fig S1 (evidence for the abnormal control replicate pair) and Table S2 (results with all samples kept).
Run 02 and 03 first.

Inputs: 图表/FigS1_异常平行/MG_PCoA_species.csv (species-level Bray-Curtis PCoA scores),
        图表/FigS1_异常平行/MG_genus_per_sample.csv (genus relative abundance per sample, %),
        宏基因组/平台导出/多样性指数结果_nr_Origin.csv (metagenome alpha diversity)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from common import FIG, GROUP_COL, INK, MG, ML_PER_AREA, MUTED, ROOT, VFA_COL, VFA_ORDER, apply_style, save, tag, \
    twoway_interaction_p

D = FIG / "FigS1_异常平行"
T = [0, 19, 31, 42, 53, 64, 78]

# gas bottles of the control and NaCl groups (raw GC areas)
ws = openpyxl.load_workbook(ROOT / "水热炭解抑制产氢.xlsx", data_only=True)["AH对照"]
gas = {}
for r in range(1, 45):
    s = ws.cell(r, 4).value
    if isinstance(s, str) and s[:2] in ("A0", "AY") and s not in gas:
        gas[s] = [0.0] + [ws.cell(r, 5 + k).value * ML_PER_AREA for k in range(6)]
gas = pd.DataFrame(gas, index=T)

# sensitivity: statistics with and without the abnormal pair
fits = pd.read_csv(FIG / "Fig3" / "TableS4_Gompertz_relief.csv", encoding="utf-8-sig")
fits_all = pd.concat([fits, pd.DataFrame([{"Sample": "A0-1", "Group": "A0", "H2_final_mL": gas["A0-1"].iloc[-1],
                                          "H2_peak_mL": gas["A0-1"].max()}])], ignore_index=True)
alpha = pd.read_csv(MG / "平台导出" / "多样性指数结果_nr_Origin.csv"); alpha["g"] = alpha.Sample.str[:2]
env = pd.read_csv(ROOT / "环境因子表_宏基因组终点.csv", encoding="utf-8-sig"); env["g"] = env.Sample.str[:2]


def ttest(df, col, drop):
    a0 = df[(df.g == "A0") & (~df.Sample.isin(drop))][col]
    return stats.ttest_ind(df[df.g == "AY"][col], a0).pvalue


a0_all, a0_ex, ay = fits_all[fits_all.Group == "A0"].H2_final_mL, fits[fits.Group == "A0"].H2_final_mL, \
    fits[fits.Group == "AY"].H2_final_mL
sens = {"A0_H2_final": (a0_all.mean(), a0_all.std(), a0_ex.mean(), a0_ex.std()),
        "retention_none": (ay.mean() / a0_all.mean(), ay.mean() / a0_ex.mean())}
for key, col, add in [("p_int_Ni_final", "H2_final_mL", "M"), ("p_int_Ni_peak", "H2_peak_mL", "M"),
                      ("p_int_HC_final", "H2_final_mL", "H")]:
    sens[key] = (twoway_interaction_p(fits_all.Group, fits_all[col], add), twoway_interaction_p(fits.Group, fits[col], add))
sens["p_Shannon_AY_vs_A0"] = (ttest(alpha, "shannon", []), ttest(alpha, "shannon", ["A0_1"]))
sens["p_butyrate_AY_vs_A0"] = (ttest(env, "Butyrate_pct", []), ttest(env, "Butyrate_pct", ["A0_1"]))
json.dump(sens, open(D / "TableS2_sensitivity.json", "w", encoding="utf-8"), indent=1, default=float)

# figure
apply_style()
fig = plt.figure(figsize=(7.1, 5.8))
gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.42, left=0.08, right=0.84, top=0.96, bottom=0.2)

ax = fig.add_subplot(gs[0, 0]); tag(ax, "A", x=-0.24)
ays = gas[[c for c in gas if c.startswith("AY")]]
ax.fill_between(T, ays.mean(axis=1) - ays.std(axis=1), ays.mean(axis=1) + ays.std(axis=1), color=GROUP_COL["AY"], alpha=0.18, lw=0)
ax.plot(T, ays.mean(axis=1), color=GROUP_COL["AY"], ls="--", lw=1.2, label="NaCl (mean ± SD)")
for s, mk in [("A0-2", "s"), ("A0-3", "^"), ("A0-1", "o")]:
    odd = s == "A0-1"
    ax.plot(T, gas[s], color=GROUP_COL["A0"], lw=1.2, marker=mk, ms=3.6, mfc="white" if odd else GROUP_COL["A0"],
            mec=GROUP_COL["A0"], label=f"Control bottle {s[-1]}" + (" (excluded)" if odd else ""))
ax.set_xticks(T); ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative H$_2$ (mL)"); ax.set_ylim(0, 45)
ax.legend(frameon=False, loc="upper left", handlelength=1.8, labelspacing=0.3)

ax = fig.add_subplot(gs[0, 1]); tag(ax, "B", x=-0.24)
pc = pd.read_csv(D / "MG_PCoA_species.csv")
for grp, d in pc.groupby("group"):
    ax.scatter(d.PC1, d.PC2, s=24, color=GROUP_COL[grp], edgecolor=INK, linewidth=0.4, zorder=3)
odd = pc[(pc.group == "A0") & (pc.PC1 > 0.3)].iloc[0]
ax.scatter(odd.PC1, odd.PC2, s=90, facecolor="none", edgecolor=INK, linewidth=0.9, zorder=4)
ax.annotate("A0_1", xy=(odd.PC1, odd.PC2), xytext=(odd.PC1 - 0.32, odd.PC2 - 0.22), fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
ax.axhline(0, color=MUTED, lw=0.6, ls=":"); ax.axvline(0, color=MUTED, lw=0.6, ls=":")
ax.set_xlabel("PC1 (52.35%)"); ax.set_ylabel("PC2 (24.53%)"); ax.set_xlim(-0.5, 0.8); ax.set_ylim(-0.6, 0.45)
labels = {"A0": "Control", "AY": "NaCl", "H0": "Hydrochar", "M0": "Ni", "MY": "Ni + NaCl"}
ax.legend(handles=[Line2D([], [], marker="o", lw=0, color=GROUP_COL[k], mec=INK, ms=4.5, label=v) for k, v in labels.items()],
          frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), handletextpad=0.2)

smp = ["A0_1", "A0_2", "A0_3", "AY_1", "AY_2", "AY_3"]
xs = np.arange(len(smp))
ax = fig.add_subplot(gs[1, 0]); tag(ax, "C", x=-0.24)
gen = pd.read_csv(D / "MG_genus_per_sample.csv", index_col=0, encoding="utf-8-sig")
show = ["Thermoanaerobacterium", "Clostridium", "Schnuerera", "Lactobacillus", "Heyndrickxia", "Caldifermentibacillus"]
cols = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
g6 = gen.loc[show, smp]; g6.loc["Others"] = 100 - g6.sum()
bottom = np.zeros(len(smp))
for name, c in zip(show + ["Others"], cols + ["#c3c2b7"]):
    ax.bar(xs, g6.loc[name], bottom=bottom, width=0.62, color=c, edgecolor=INK, linewidth=0.4,
           label=name if name == "Others" else f"$\\it{{{name}}}$")
    bottom += g6.loc[name].values
ax.set_xticks(xs, smp, rotation=30, ha="right"); ax.set_ylabel("Relative abundance (%)"); ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, labelspacing=0.25, handlelength=1.0,
          columnspacing=0.8)

ax = fig.add_subplot(gs[1, 1]); tag(ax, "D", x=-0.24)
e = env.set_index("Sample").loc[smp]
bottom = np.zeros(len(smp))
for name in VFA_ORDER:
    v = e[name + "_mgCOD"].values
    ax.bar(xs, v, bottom=bottom, width=0.62, color=VFA_COL[name], edgecolor=INK, linewidth=0.4, label=name)
    bottom += v
for x, b, pct in zip(xs, bottom, e.Butyrate_pct.values):
    ax.text(x, b + 90, f"{pct:.0f}%", ha="center", fontsize=6.5)
ax.set_xticks(xs, smp, rotation=30, ha="right"); ax.set_ylabel("Products at end (mg COD/L)"); ax.set_ylim(0, 6000)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, labelspacing=0.25, handlelength=1.0,
          columnspacing=0.8)
save(fig, D / "FigS1")
print({k: [round(x, 4) for x in v] for k, v in sens.items()})
