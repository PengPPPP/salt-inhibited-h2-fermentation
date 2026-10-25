"""Fig S3: liquid products, soluble carbohydrate and protein over time in the relief batches (A0_1 excluded).

Liquid samples were taken at the 1st, 3rd, 4th and 6th gas measurements: 19/42/53/78 h (control and hydrochar batch)
and 20/46/58/80 h (Ni batch). VFA workbooks, Sheet2: one block per sampling time (time label in column B), sample
names in column C and COD values (mg/L) in columns Y-AD. Carbohydrate / protein workbooks: concentrations (g/L) of
the four sampling times in columns H-K of the first sheet, diluted 4x.
"""
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.lines import Line2D

from common import DILUTION, FIG, GROUP_COL, GROUP_LABEL, INK, ROOT, VFA, apply_style, save, tag

D = FIG / "FigS3"
D.mkdir(parents=True, exist_ok=True)
GROUPS = ["A0", "AY", "H0", "HY", "M0", "MY"]
EXCLUDE = ["A0-1"]
BATCH = {"水热炭解抑制VFA.xlsx": ([47, 77, 107, 137], [19, 42, 53, 78]),
         "活性炭氯化镍解抑制VFA.xlsx": ([43, 73, 103, 133], [20, 46, 58, 80])}
TIMES = {"A": [19, 42, 53, 78], "H": [19, 42, 53, 78], "M": [20, 46, 58, 80]}


def wb(name):
    return openpyxl.load_workbook(ROOT / name, data_only=True)


rows = []
for name, (starts, times) in BATCH.items():
    ws = wb(name)["Sheet2"]
    for r0, t in zip(starts, times):
        for r in range(r0, r0 + 30):
            s = ws.cell(r, 3).value
            if isinstance(s, str) and s[:2] in GROUPS and s not in EXCLUDE:
                rows.append({"Sample": s, "Group": s[:2], "Time": t, **{v: ws.cell(r, 25 + j).value for j, v in enumerate(VFA)}})
chem = pd.DataFrame(rows).drop_duplicates(["Sample", "Time"]).set_index(["Sample", "Time"])

for name in ["水热炭解抑制碳水.xlsx", "活性炭氯化镍 碳水.xlsx", "水热碳解抑制蛋白质.xlsx", "活性炭氯化镍 蛋白质.xlsx"]:
    key = "Carbohydrate" if "碳水" in name else "Protein"
    ws = wb(name).worksheets[0]
    seen = set()
    for r in range(1, ws.max_row + 1):
        s = ws.cell(r, 2).value
        vals = [ws.cell(r, c).value for c in range(8, 12)]
        if s in chem.index.get_level_values(0) and s not in seen and all(isinstance(v, (int, float)) for v in vals):
            seen.add(s)
            for t, v in zip(TIMES[s[0]], vals):
                chem.loc[(s, t), key] = v * 1000 * DILUTION
chem = chem.reset_index()
chem["TotalVFA"] = chem[VFA].sum(axis=1)
chem.to_csv(D / "figS3_relief_timecourse.csv", index=False, encoding="utf-8-sig")

# endpoint check against the Fig 3 table
end = pd.read_csv(FIG / "Fig3" / "fig3_endpoint_chem.csv", encoding="utf-8-sig").set_index("Sample")
last = chem.sort_values("Time").groupby("Sample").last()
for c in ["Butyrate", "Acetate", "Carbohydrate", "Protein"]:
    assert np.allclose(last.loc[end.index, c], end[c], atol=0.05), c

apply_style()
fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.1))
fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.11, wspace=0.42, hspace=0.5)
PANELS = [("TotalVFA", "Total products (mg COD/L)"), ("Butyrate", "Butyrate (mg COD/L)"),
          ("Acetate", "Acetate (mg COD/L)"), ("Ethanol", "Ethanol (mg COD/L)"),
          ("Carbohydrate", "Soluble carbohydrate (mg/L)"), ("Protein", "Soluble protein (mg/L)")]
for k, (ax, (col, lab)) in enumerate(zip(axs.flat, PANELS)):
    tag(ax, "ABCDEF"[k], x=-0.3)
    for g in GROUPS:
        sub = chem[chem.Group == g].groupby("Time")[col]
        salt = g.endswith("Y")
        ax.errorbar(sub.mean().index, sub.mean(), yerr=sub.std().fillna(0), fmt="-o", ms=3.0, lw=0.9,
                    ls="--" if salt else "-", color=GROUP_COL[g], mfc="white" if salt else GROUP_COL[g],
                    mec=GROUP_COL[g], mew=0.7, elinewidth=0.5, capsize=1.2)
    ax.set_xticks([0, 20, 40, 60, 80]); ax.set_xlim(10, 86)
    ax.set_xlabel("Time (h)"); ax.set_ylabel(lab); ax.set_ylim(bottom=0)
handles = [Line2D([], [], color=GROUP_COL[g], marker="o", ms=3.0, lw=0.9, ls="--" if g.endswith("Y") else "-",
                  mfc="white" if g.endswith("Y") else GROUP_COL[g], mec=GROUP_COL[g], label=GROUP_LABEL[g])
           for g in GROUPS]
fig.legend(handles=handles, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6.6,
           handlelength=2.0, columnspacing=1.2)
save(fig, D / "FigS3")
print(chem.groupby(["Group", "Time"])[["Butyrate", "Acetate", "TotalVFA", "Carbohydrate"]].mean().round(0).to_string())
