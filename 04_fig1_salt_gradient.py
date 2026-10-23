"""Fig 1 (salt-gradient H2 production and metabolites) and Fig S2 (time courses). Run 01 first."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from common import FIG, INK, NACL, RAMP, SALT_GROUPS, VFA_COL, VFA_ORDER, apply_style, gompertz, save, tag

D = FIG / "Fig1"
h2 = pd.read_csv(D / "fig1_h2_timecourse.csv", index_col=0, encoding="utf-8-sig")
h2.columns = h2.columns.astype(int)
chem = pd.read_csv(D / "fig1_chem_timecourse.csv", encoding="utf-8-sig")
fits = pd.read_csv(D / "TableS3_Gompertz_Y.csv", encoding="utf-8-sig")
st = json.load(open(D / "fig1_stats.json", encoding="utf-8"))
apply_style()
handles = [Line2D([], [], color=RAMP[g], marker="o", ms=3.2, mec="white", mew=0.4, lw=1.2, label=str(NACL[g]))
           for g in SALT_GROUPS]


def regline(ax, x, y):
    r = stats.linregress(x, y)
    xs = np.linspace(0, 20, 50)
    ax.plot(xs, r.intercept + r.slope * xs, color=INK, lw=0.8, ls="--", zorder=1)
    return f"r = {r.rvalue:.2f}, " + ("P < 0.001" if r.pvalue < 0.001 else f"P = {r.pvalue:.3f}")


def dots(ax, col):
    for g in SALT_GROUPS:
        v = fits[fits.Group == g][col].values
        ax.scatter(NACL[g] + np.array([-0.6, 0, 0.6]), v, s=9, color=RAMP[g], edgecolor="white", linewidth=0.4, zorder=3)
        ax.errorbar(NACL[g], v.mean(), yerr=v.std(ddof=1), fmt="_", color=INK, ms=7, mew=1.0, elinewidth=0.7,
                    capsize=1.8, zorder=4)
    ax.set_xticks([0, 5, 10, 15, 20]); ax.set_xlim(-2.5, 22.5); ax.set_xlabel("NaCl (g/L)")


fig = plt.figure(figsize=(7.1, 4.7))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42, left=0.07, right=0.985, top=0.95, bottom=0.17)

ax = fig.add_subplot(gs[0, 0]); tag(ax, "A")
tt = np.linspace(0, 67, 200)
par = fits.set_index("Sample")[["Hmax_mL", "Rm_mL_h", "lambda_h"]]
for g in SALT_GROUPS:
    reps = [s for s in h2.index if s.startswith(g)]
    ax.plot(tt, np.mean([gompertz(tt, *par.loc[s]) for s in reps], axis=0), color=RAMP[g], lw=1.2, zorder=2)
    ax.errorbar(h2.columns, h2.loc[reps].mean(), yerr=h2.loc[reps].std(), fmt="o", ms=3.2, color=RAMP[g],
                mec="white", mew=0.4, elinewidth=0.6, capsize=1.5, zorder=3)
ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative H$_2$ (mL)")
ax.set_xticks([0, 15, 30, 45, 60]); ax.set_xlim(-2, 70); ax.set_ylim(0, 62)
ax.legend(handles=handles, title="NaCl (g/L)", title_fontsize=6.5, frameon=False, loc="upper left",
          handlelength=1.4, borderaxespad=0.1, labelspacing=0.25)

ax = fig.add_subplot(gs[0, 1]); tag(ax, "B")
dots(ax, "H2_67h_mL")
txt = regline(ax, fits.NaCl, fits.H2_67h_mL)
for g in SALT_GROUPS:
    ax.text(NACL[g], fits[fits.Group == g].H2_67h_mL.max() + 3.2, st["H2_67h"]["letters"][g], ha="center", fontsize=6.5)
    if g != "Y0":
        ax.text(NACL[g], 2.5, f"−{st['inhibition_pct'][g]:.0f}%", ha="center", fontsize=6)
ax.set_ylabel("H$_2$ at 67 h (mL)"); ax.set_ylim(0, 66)
ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6)

ax = fig.add_subplot(gs[0, 2]); tag(ax, "C")
dots(ax, "lambda_h"); txt = regline(ax, fits.NaCl, fits.lambda_h)
ax.set_ylabel("Lag phase λ (h)"); ax.set_ylim(0, 32)
ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=6)

ax = fig.add_subplot(gs[1, 0]); tag(ax, "D")
dots(ax, "Rm_mL_h"); txt = regline(ax, fits.NaCl, fits.Rm_mL_h)
ax.set_ylabel("Max. H$_2$ rate R$_m$ (mL/h)"); ax.set_ylim(0, 1.25)
ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6)

ax = fig.add_subplot(gs[1, 1]); tag(ax, "E")
end = chem[chem.Time == 67].groupby("Group")[VFA_ORDER].mean().loc[SALT_GROUPS]
bottom = np.zeros(5); x = np.arange(5)
for comp in VFA_ORDER:
    ax.bar(x, end[comp], bottom=bottom, width=0.62, color=VFA_COL[comp], edgecolor=INK, linewidth=0.4, label=comp, zorder=2)
    bottom += end[comp].values
for i, g in enumerate(SALT_GROUPS):
    ax.text(i, bottom[i] + 150, f"{st['Butyrate_pct_67h']['mean'][g]:.0f}%", ha="center", fontsize=6)
ax.text(1.0, 1.0, "Butyrate share above bars", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
ax.set_xticks(x, [str(NACL[g]) for g in SALT_GROUPS]); ax.set_xlabel("NaCl (g/L)")
ax.set_ylabel("Products at 67 h (mg COD/L)"); ax.set_ylim(0, 9200)
ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=0.9,
          columnspacing=0.9, handletextpad=0.4, labelspacing=0.3)

ax = fig.add_subplot(gs[1, 2]); tag(ax, "F")
for g in SALT_GROUPS:
    sub = chem[chem.Group == g].groupby("Time").Carbohydrate
    ax.errorbar(sub.mean().index, sub.mean(), yerr=sub.std(), fmt="-o", ms=3.2, color=RAMP[g], mec="white", mew=0.4,
                lw=1.0, elinewidth=0.6, capsize=1.5)
ax.set_xlabel("Time (h)"); ax.set_ylabel("Soluble carbohydrate (mg/L)")
ax.set_xticks([15, 36, 46, 67]); ax.set_xlim(10, 72); ax.set_ylim(150, 980)
ax.legend(handles=handles, title="NaCl (g/L)", title_fontsize=6.5, frameon=False, loc="upper right", ncol=5,
          handlelength=1.0, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.0)
save(fig, D / "Fig1", tif=True)

# Fig S2
figS, axs = plt.subplots(1, 4, figsize=(7.1, 1.95))
figS.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.22, wspace=0.55)
for k, (ax, (col, lab)) in enumerate(zip(axs, [("TotalVFA", "Total products (mg COD/L)"), ("Butyrate", "Butyrate (mg COD/L)"),
                                               ("Acetate", "Acetate (mg COD/L)"), ("Protein", "Soluble protein (mg/L)")])):
    tag(ax, "ABCD"[k])
    for g in SALT_GROUPS:
        sub = chem[chem.Group == g].groupby("Time")[col]
        ax.errorbar(sub.mean().index, sub.mean(), yerr=sub.std(), fmt="-o", ms=2.8, color=RAMP[g], mec="white",
                    mew=0.4, lw=0.9, elinewidth=0.5, capsize=1.2)
    ax.set_xticks([15, 36, 46, 67]); ax.set_xlabel("Time (h)"); ax.set_ylabel(lab); ax.set_ylim(bottom=0)
figS.legend(handles=handles, title="NaCl (g/L)", title_fontsize=6, fontsize=6, frameon=False, ncol=6,
            loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=0.8)
save(figS, D / "FigS2")
print("Fig1, FigS2 saved")
