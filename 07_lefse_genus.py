"""Genus-level LEfSe bar chart (LDA >= 3.5) from the platform LEfSe result table (Bacteria, phylum-to-genus)."""
import matplotlib.pyplot as plt
import pandas as pd

from common import GROUP_COL, INK, MG, apply_style

LDA_MIN = 3.5
ORDER = ["A0", "AY", "H0", "M0", "MY"]
D = MG / "物种与功能差异分析" / "LEFSe差异辨别分析_细菌"
src = next(D.glob("lefse判别结果表_*.csv"))

d = pd.read_csv(src)
d = d[d.Pvalue.astype(str) != "-"].copy()
d["LDA"] = d.LDA_value.astype(float)
d["P"] = d.Pvalue.astype(float)
d["taxon"] = d["Species name"].str.split(".").str[-1]
g = d[d.taxon.str.startswith("g__") & (d.LDA >= LDA_MIN)].copy()
g["genus"] = g.taxon.str[3:].str.replace("_f__", " (f. ", regex=False)
g.loc[g.genus.str.contains(r"\(f\. "), "genus"] += ")"
g["rank"] = g.Group.map({k: i for i, k in enumerate(ORDER)})
g = g.sort_values(["rank", "LDA"], ascending=[True, False])
g[["Group", "genus", "LDA", "P", "Mean"]].to_csv(D / "LEfSe_genus_LDA3.5.csv", index=False, encoding="utf-8-sig")

apply_style()
rows, y = [], 0
for grp in ORDER:
    sub = g[g.Group == grp]
    if sub.empty:
        continue
    rows.append(("header", grp, None, y)); y += 1
    for _, r in sub.iterrows():
        rows.append(("bar", grp, r, y)); y += 1
    y += 0.4
fig, ax = plt.subplots(figsize=(4.6, 0.26 * y + 0.6))
for kind, grp, r, yy in rows:
    if kind == "header":
        ax.text(0, yy, f"Enriched in {grp}", fontweight="bold", va="center", ha="left")
        continue
    ax.barh(yy, r.LDA, height=0.62, color=GROUP_COL[grp], edgecolor=INK, linewidth=0.4)
    ax.text(r.LDA + 0.05, yy, f"{r.LDA:.2f}", va="center", ha="left", fontsize=7)
    ax.text(-0.08, yy, r.genus, va="center", ha="right", style="italic", fontsize=7.5)
ax.set_ylim(y - 0.4, -0.8); ax.set_xlim(0, 6.2); ax.set_yticks([])
ax.set_xlabel("LDA score (log$_{10}$)")
fig.subplots_adjust(left=0.42, right=0.97, top=0.98, bottom=0.08)
for ext in ("png", "pdf"):
    fig.savefig(D / f"LEfSe_genus_LDA{LDA_MIN}.{ext}")
print(g[["Group", "genus", "LDA"]].to_string(index=False))
