"""Metagenome functional genes of the relief batch (KEGG): key fermentation, H2, osmoadaptation and Ni-related genes.
Abundance = KO read counts per million KO-annotated reads (CPM); multi-subunit complexes are the mean of their
subunits. Group means, log2 fold changes of NaCl (without / with Ni) and Ni x NaCl interaction (2 x 2 ANOVA on
ln(CPM + 1)). Main analysis without A0_1.

Inputs: 宏基因组/物种与通路功能数据库/KEGG功能注释/<sample>/kegg_KO_profile.xls
"""
import json

import numpy as np
import pandas as pd

from common import FIG, MG, twoway_interaction_p

D = FIG / "Fig5"
D.mkdir(parents=True, exist_ok=True)
GROUPS = ["A0", "AY", "H0", "M0", "MY"]
EXCLUDE = ["A0_1"]
K = lambda a, b: [f"K{i:05d}" for i in range(a, b + 1)]

# key, KOs, aggregation of the KOs ("mean" for subunits of one complex, "sum" for alternative enzymes)
GENES = {
    # pyruvate oxidation and fermentation branches
    "nifJ": (["K03737"], "sum"), "porABCD": (K(169, 172), "mean"), "pflB": (["K00656"], "sum"),
    "ldh": (["K00016"], "sum"), "ldhA": (["K03778"], "sum"), "adhE": (["K04072"], "sum"),
    "pta": (["K00625"], "sum"), "ackA": (["K00925"], "sum"),
    # butyrate pathway
    "thl": (["K00626"], "sum"), "hbd": (["K00074"], "sum"), "crt": (["K01715", "K17865"], "sum"),
    "bcd": (["K00248"], "sum"), "etfAB": (["K03521", "K03522"], "mean"),
    "ptb": (["K00634"], "sum"), "buk": (["K00929"], "sum"), "atoDA": (["K01034", "K01035"], "mean"),
    # hydrogenases
    "hndD": (["K18332"], "sum"), "echA-F": (K(14086, 14091), "mean"),
    # osmoadaptation
    "mrpA-G": (K(5565, 5571), "mean"), "nhaC": (["K03315"], "sum"), "proVWX": (K(2000, 2002), "mean"),
    "opuA-C": (K(5845, 5847), "mean"), "opuD": (["K05020"], "sum"), "kdpABC": (K(1546, 1548), "mean"),
    # Ni/Co uptake, Ni-responsive regulation, [NiFe]-hydrogenase maturation and the Ni enzyme CODH
    "cbiMNOQ": (K(2006, 2009), "mean"), "nikR": (["K07722"], "sum"), "hypA-F": (K(4651, 4656), "mean"),
    "cooS": (["K00198"], "sum"),
}

cols = {}
for f in sorted((MG / "物种与通路功能数据库" / "KEGG功能注释").glob("*/kegg_KO_profile.xls")):
    t = pd.read_csv(f, sep="\t")
    cols[t.columns[1]] = t.set_index("#KO")[t.columns[1]]
ko = pd.DataFrame(cols).fillna(0)
samples = [s for g in GROUPS for s in sorted(ko.columns) if s.startswith(g) and s not in EXCLUDE]
cpm = ko[samples] / ko[samples].sum() * 1e6
genes = pd.DataFrame({k: getattr(cpm.reindex(kos).fillna(0), how)() for k, (kos, how) in GENES.items()}).T
genes.to_csv(D / "fig5_genes_cpm_per_sample.csv", encoding="utf-8-sig")
labels = np.array([s[:2] for s in samples])

out = {}
for g, row in genes.iterrows():
    m = {grp: float(row[labels == grp].mean()) for grp in GROUPS}
    out[g] = {"mean": m, "sd": {grp: float(row[labels == grp].std(ddof=1)) for grp in GROUPS},
              "log2fc_noNi": float(np.log2((m["AY"] + 1) / (m["A0"] + 1))),
              "log2fc_Ni": float(np.log2((m["MY"] + 1) / (m["M0"] + 1))),
              "ratio_noNi": m["AY"] / m["A0"], "ratio_Ni": m["MY"] / m["M0"],
              "p_int_Ni": float(twoway_interaction_p(labels, row.values + 1, "M"))}
json.dump(out, open(D / "fig5_gene_stats.json", "w", encoding="utf-8"), indent=1)
for g, s in out.items():
    print(f"{g:8s}", {k: round(v) for k, v in s["mean"].items()},
          f"AY/A0={s['ratio_noNi']:.2f} MY/M0={s['ratio_Ni']:.2f} P_int={s['p_int_Ni']:.4f}")
