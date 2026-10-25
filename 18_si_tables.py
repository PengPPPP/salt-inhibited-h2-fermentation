"""Supplementary tables built from the analysis outputs (run 02, 10, 12 and 13 first).

Table S5  modified Gompertz parameters of the relief batches (Tukey letters for Rm and lambda)
Table S6  multivariate tests: PERMANOVA (five groups; NaCl x Ni) and db-RDA (VIF, marginal tests)
Table S7  key genera and Shannon index: mean +/- SD, Tukey letters, Ni x NaCl interaction
Table S9  functional gene abundance (CPM): mean +/- SD and Ni x NaCl interaction
"""
import json

import pandas as pd

from common import FIG, compact_letters

D = FIG / "SI_tables"
D.mkdir(parents=True, exist_ok=True)
GROUPS = ["A0", "AY", "H0", "M0", "MY"]
pf = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
ms = lambda m, s, d=1: f"{m:.{d}f} ± {s:.{d}f}"

# S5: Gompertz parameters, relief batches
fits = pd.read_csv(FIG / "Fig3" / "TableS4_Gompertz_relief.csv", encoding="utf-8-sig")
order = ["A0", "AY", "H0", "HY", "M0", "MY"]
lett = {c: compact_letters(order, [fits[fits.Group == g][c].values for g in order]) for c in ["Rm_mL_h", "lambda_h"]}
rows = []
for g in order:
    f = fits[fits.Group == g]
    rows.append({"Group": g, "n": len(f), "Hmax": ms(f.Hmax_mL.mean(), f.Hmax_mL.std()),
                 "Rm": f"{ms(f.Rm_mL_h.mean(), f.Rm_mL_h.std(), 2)} {lett['Rm_mL_h'][g]}",
                 "lambda": f"{ms(f.lambda_h.mean(), f.lambda_h.std())} {lett['lambda_h'][g]}",
                 "R2": f"{f.R2.min():.3f}–{f.R2.max():.3f}"})
pd.DataFrame(rows).to_csv(D / "TableS5_Gompertz_relief.csv", index=False, encoding="utf-8-sig")

# S6: PERMANOVA and db-RDA
f4 = json.load(open(FIG / "Fig4" / "fig4_stats.json", encoding="utf-8"))
f6 = json.load(open(FIG / "Fig6" / "fig6_stats.json", encoding="utf-8"))
rows = [{"Analysis": "PERMANOVA（五组）", "Term": "处理", "VIF": "", **{k: v for k, v in f4["permanova_5groups"]["group"].items()}}]
rows += [{"Analysis": "PERMANOVA（NaCl × 镍）", "Term": t.replace("x", "×"), "VIF": "", **f4["permanova_2x2"][t]}
         for t in ["NaCl", "Ni", "NaCl x Ni"]]
rows += [{"Analysis": "db-RDA", "Term": "全模型", "VIF": "", "R2": f6["R2"], "F": f6["F"], "P": f6["P"]}]
rows += [{"Analysis": "db-RDA（边际检验）", "Term": t, "VIF": f"{f6['vif'][t]:.2f}", **f6["margin"][t]} for t in f6["constraints"]]
t6 = pd.DataFrame(rows)
t6["R2"] = t6.R2.map(lambda v: f"{v:.3f}"); t6["F"] = t6.F.map(lambda v: f"{v:.2f}"); t6["P"] = t6.P.map(pf)
t6.to_csv(D / "TableS6_multivariate.csv", index=False, encoding="utf-8-sig")
with open(D / "TableS6_note.txt", "w", encoding="utf-8") as fh:
    fh.write(f"{f6['R2_adj']:.3f}")

# S7: key genera and Shannon
rows = []
for name in ["Clostridium", "Schnuerera", "Thermoanaerobacterium", "Lactobacillus", "Caldifermentibacillus",
             "Caldibacillus", "Garciella", "Neomoorella", "Symbiobacterium", "Shannon"]:
    s = f4[name]
    d = 2 if name == "Shannon" else 1
    rows.append({"Measure": name, **{g: f"{ms(s['mean'][g], s['sd'][g], d)} {s['letters'][g]}" for g in GROUPS},
                 "P_int": pf(s["p_int_Ni"])})
pd.DataFrame(rows).to_csv(D / "TableS7_genera.csv", index=False, encoding="utf-8-sig")

# S9: functional genes
f5 = json.load(open(FIG / "Fig5" / "fig5_gene_stats.json", encoding="utf-8"))
ORDER = ["nifJ", "porABCD", "pflB", "ldh", "ldhA", "adhE", "pta", "ackA", "thl", "hbd", "crt", "bcd", "etfAB", "ptb",
         "buk", "atoDA", "hndD", "echA-F", "mrpA-G", "nhaC", "proVWX", "opuA-C", "opuD", "cbiMNOQ", "nikR", "hypA-F"]
rows = [{"Gene": g, **{k: ms(f5[g]["mean"][k], f5[g]["sd"][k], 0) for k in GROUPS}, "P_int": pf(f5[g]["p_int_Ni"])}
        for g in ORDER]
pd.DataFrame(rows).to_csv(D / "TableS9_genes.csv", index=False, encoding="utf-8-sig")
print(pd.read_csv(D / "TableS5_Gompertz_relief.csv").to_string()); print(t6.to_string())
