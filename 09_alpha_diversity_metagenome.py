"""Metagenome alpha diversity (NR species level): group means and pairwise t tests, main analysis without A0_1."""
import pandas as pd
from scipy import stats

from common import MG

GROUPS = ["A0", "AY", "H0", "M0", "MY"]
PAIRS = [("AY", "A0"), ("MY", "AY"), ("M0", "A0"), ("H0", "A0"), ("MY", "M0")]

a = pd.read_csv(MG / "平台导出" / "多样性指数结果_nr_Origin.csv")
a["Group"] = a.Sample.str[:2]
a = a[a.Group.isin(GROUPS) & (a.Sample != "A0_1")]

rows = []
for m in ["sobs", "chao", "shannon", "simpson"]:
    g = a.groupby("Group")[m]
    row = {"index": m, **{f"{k}_mean": v for k, v in g.mean().round(4).items()},
           **{f"{k}_sd": v for k, v in g.std().round(4).items()},
           "p_anova": stats.f_oneway(*[a[a.Group == k][m] for k in GROUPS]).pvalue}
    for x, y in PAIRS:
        row[f"p_{x}_vs_{y}"] = stats.ttest_ind(a[a.Group == x][m], a[a.Group == y][m]).pvalue
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(MG / "alpha_diversity_stats.csv", index=False, encoding="utf-8-sig")
print(out.round(4).T.to_string())
