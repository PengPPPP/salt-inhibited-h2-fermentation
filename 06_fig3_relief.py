"""Fig 3: relief of NaCl inhibition by hydrochar and Ni. Run 02 first."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from common import FIG, GROUP_COL, GROUP_LABEL, INK, MUTED, VFA_COL, VFA_ORDER, apply_style, pfmt, save, tag

D = FIG / "Fig3"
h2 = pd.read_csv(D / "fig3_h2_timecourse.csv", encoding="utf-8-sig")
chem = pd.read_csv(D / "fig3_endpoint_chem.csv", encoding="utf-8-sig")
fits = pd.read_csv(D / "TableS4_Gompertz_relief.csv", encoding="utf-8-sig")
st = json.load(open(D / "fig3_stats.json", encoding="utf-8"))
apply_style()
ORDER = ["A0", "AY", "H0", "HY", "M0", "MY"]
XPOS = [0, 0.8, 2.2, 3.0, 4.4, 5.2]


def timecourse(ax, groups, xmax, ticks):
    for g in groups:
        d = h2[h2.Group == g].groupby("Time").H2
        salt = g.endswith("Y")
        ax.errorbar(d.mean().index, d.mean(), yerr=d.std().fillna(0), color=GROUP_COL[g], ls="--" if salt else "-",
                    lw=1.2, marker="o", ms=3.6, mfc="white" if salt else GROUP_COL[g], mec=GROUP_COL[g], mew=1.0,
                    elinewidth=0.6, capsize=1.5, label=GROUP_LABEL[g])
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative H$_2$ (mL)")
    ax.set_xlim(-3, xmax + 3); ax.set_ylim(0, 50); ax.set_xticks(ticks)
    ax.legend(frameon=False, loc="upper left", handlelength=2.0, labelspacing=0.3, borderaxespad=0.2)


fig = plt.figure(figsize=(7.1, 4.9))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.45, left=0.075, right=0.985, top=0.95, bottom=0.13)

ax = fig.add_subplot(gs[0, 0]); tag(ax, "A"); timecourse(ax, ["A0", "AY", "H0", "HY"], 78, [0, 19, 31, 42, 53, 64, 78])
ax = fig.add_subplot(gs[0, 1]); tag(ax, "B"); timecourse(ax, ["M0", "MY"], 80, [0, 20, 35, 46, 58, 69, 80])
pk = h2[h2.Group == "MY"].groupby("Time").H2.mean()
ax.annotate("peak", xy=(pk.idxmax(), pk.max()), xytext=(pk.idxmax() + 12, pk.max() + 8), fontsize=6.5,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))

ax = fig.add_subplot(gs[0, 2]); tag(ax, "C")
for x, g in zip(XPOS, ORDER):
    v = fits[fits.Group == g].H2_final_mL.values
    salt = g.endswith("Y")
    ax.bar(x, v.mean(), width=0.7, color="white" if salt else GROUP_COL[g], edgecolor=GROUP_COL[g], linewidth=1.2,
           hatch="////" if salt else None, zorder=2)
    ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color=INK, capsize=2, elinewidth=0.7, zorder=3)
    ax.scatter(x + np.linspace(-0.15, 0.15, len(v)), v, s=7, color=INK, zorder=4)
ax.set_xticks([0.4, 2.6, 4.8], ["None", "Hydrochar", "Ni"]); ax.set_ylabel("H$_2$ at end (mL)"); ax.set_ylim(0, 60)
ax.legend(handles=[Patch(facecolor="#c3c2b7", edgecolor=INK, label="0 g/L NaCl"),
                   Patch(facecolor="white", edgecolor=INK, hatch="////", label="20 g/L NaCl")],
          frameon=False, loc="upper center", fontsize=6.5, handlelength=1.2, borderaxespad=0.2, ncol=2, columnspacing=0.8)

ax = fig.add_subplot(gs[1, 0]); tag(ax, "D")
for i, (a, name) in enumerate([("A", "None"), ("H", "Hydrochar"), ("M", "Ni")]):
    r = np.array(st["retention_H2_final_mL"][a]["values"])
    rb = chem[chem.Group == a + "Y"].Butyrate_pct.values / chem[chem.Group == a + "0"].Butyrate_pct.mean()
    for dx, vals, hatch in [(-0.18, r, None), (0.18, rb, "////")]:
        ax.bar(i + dx, vals.mean(), width=0.34, color=GROUP_COL[a + "0"] if hatch is None else "white",
               edgecolor=GROUP_COL[a + "0"], linewidth=1.2, hatch=hatch, zorder=2)
        ax.errorbar(i + dx, vals.mean(), yerr=vals.std(ddof=1), color=INK, capsize=2, elinewidth=0.7, zorder=3)
        ax.scatter(i + dx + np.linspace(-0.07, 0.07, len(vals)), vals, s=6, color=INK, zorder=4)
    if a != "A":
        ax.text(i, 1.06, pfmt(st[f"twoway_{a}_H2_final_mL"]["p_interaction"]), ha="center", fontsize=6.3)
ax.axhline(1, color=MUTED, lw=0.6, ls=":")
ax.set_xticks(range(3), ["None", "Hydrochar", "Ni"]); ax.set_ylim(0, 1.25); ax.set_ylabel("Relative to 0 g/L NaCl")
ax.legend(handles=[Patch(facecolor="#c3c2b7", edgecolor=INK, label="H$_2$"),
                   Patch(facecolor="white", edgecolor=INK, hatch="////", label="Butyrate share")],
          frameon=False, loc="upper left", fontsize=6.5, handlelength=1.2, borderaxespad=0.2, ncol=2)

ax = fig.add_subplot(gs[1, 1:]); tag(ax, "E", x=-0.115)
end = chem.groupby("Group")[VFA_ORDER].mean().loc[ORDER]
bottom = np.zeros(len(ORDER))
for comp in VFA_ORDER:
    ax.bar(XPOS, end[comp], bottom=bottom, width=0.62, color=VFA_COL[comp], edgecolor=INK, linewidth=0.4, label=comp, zorder=2)
    bottom += end[comp].values
bp = chem.groupby("Group").Butyrate_pct.mean()
for x, g, b in zip(XPOS, ORDER, bottom):
    ax.text(x, b + 120, f"{bp[g]:.0f}%", ha="center", fontsize=6.5)
ax.set_xticks(XPOS, [GROUP_LABEL[g] for g in ORDER], rotation=30, ha="right")
ax.set_ylabel("Products at end (mg COD/L)"); ax.set_ylim(0, 7600); ax.set_xlim(-0.6, 5.8)
ax.text(1.0, 1.01, "Butyrate share above bars", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.8, labelspacing=0.35,
          handlelength=1.0, reverse=True)
pos = ax.get_position(); ax.set_position([pos.x0, pos.y0, pos.width * 0.78, pos.height])
save(fig, D / "Fig3", tif=True)
print("Fig3 saved")
