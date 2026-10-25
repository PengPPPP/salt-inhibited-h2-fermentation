"""Metagenome community of the relief batch (NR annotation): genus table, Bray-Curtis PCoA, PERMANOVA and
2 x 2 tests (Ni x NaCl) on key genera and Shannon diversity. Main analysis without A0_1.

Inputs: 宏基因组/物种与通路功能数据库/NR物种注释/<sample>/tax_s.xls (species read counts per sample),
        宏基因组/平台导出/多样性指数结果_nr_Origin.csv (alpha diversity)
"""
import json

import numpy as np
import pandas as pd

from common import FIG, MG, compact_letters, twoway_interaction_p

D = FIG / "Fig4"
D.mkdir(parents=True, exist_ok=True)
GROUPS = ["A0", "AY", "H0", "M0", "MY"]
EXCLUDE = ["A0_1"]
N_PERM = 9999
rng = np.random.default_rng(2026)

# species read counts -> relative abundance (%)
cols = {}
for f in sorted((MG / "物种与通路功能数据库" / "NR物种注释").glob("*/tax_s.xls")):
    t = pd.read_csv(f, sep="\t")
    cols[t.columns[1]] = t.set_index("#Taxonomy")[t.columns[1]]
sp = pd.DataFrame(cols).fillna(0)
samples = [s for g in GROUPS for s in sorted(sp.columns) if s.startswith(g) and s not in EXCLUDE]
sp_rel = sp[samples] / sp[samples].sum() * 100
genus = sp.groupby(sp.index.str.split(";").str[6].str.replace("g__", "", regex=False)).sum()
genus_rel = genus / genus.sum() * 100
genus_rel.to_csv(D / "fig4_genus_per_sample.csv", encoding="utf-8-sig")     # all 15 samples
grp = pd.Series([s[:2] for s in samples], index=samples)

# Clostridium species (group means, %), species above 1% in any group
cl = sp_rel[sp_rel.index.str.contains(";g__Clostridium;")]
cl = cl.T.groupby(grp).mean().T[GROUPS]
cl.index = cl.index.str.split(";").str[-1].str.replace("s__", "", regex=False).str.replace("_", " ")
cl[cl.max(axis=1) >= 1].sort_values("MY", ascending=False).to_csv(D / "fig4_clostridium_species.csv", encoding="utf-8-sig")


# Bray-Curtis on species relative abundance, PCoA
def bray_curtis(x):
    x = x.values.T
    return np.array([[np.abs(a - b).sum() / (a + b).sum() for b in x] for a in x])


dist = bray_curtis(sp_rel)
n = len(samples)
C = np.eye(n) - 1 / n
G = C @ (-0.5 * dist ** 2) @ C
w, v = np.linalg.eigh(G)
w, v = w[::-1], v[:, ::-1]
expl = w / w[w > 0].sum()
pcoa = pd.DataFrame({"Sample": samples, "Group": grp.values, "PC1": v[:, 0] * np.sqrt(w[0]), "PC2": v[:, 1] * np.sqrt(w[1])})
pcoa.to_csv(D / "fig4_pcoa_species.csv", index=False, encoding="utf-8-sig")


def permanova(G, designs, idx_sets, n_perm=N_PERM):
    """Sequential (type I) PERMANOVA on the Gower matrix; designs are nested model matrices, free permutation."""
    def f_stats(Gp):
        tr = [np.trace(X @ np.linalg.pinv(X) @ Gp) for X in designs]
        ss = np.diff(tr)
        res = np.trace(Gp) - tr[-1]
        df = np.diff([np.linalg.matrix_rank(X) for X in designs])
        dfr = len(Gp) - np.linalg.matrix_rank(designs[-1])
        return (ss / df) / (res / dfr), ss / np.trace(Gp)
    F0, R2 = f_stats(G)
    hits = np.zeros_like(F0)
    for _ in range(n_perm):
        p = rng.permutation(len(G))
        hits += f_stats(G[np.ix_(p, p)])[0] >= F0 - 1e-12
    return {name: {"F": float(f), "R2": float(r), "P": float((h + 1) / (n_perm + 1))}
            for name, f, r, h in zip(idx_sets, F0, R2, hits)}


one = np.ones((n, 1))
dummies = pd.get_dummies(grp).values.astype(float)
stats_out = {"explained": {"PC1": float(expl[0]), "PC2": float(expl[1])},
             "permanova_5groups": permanova(G, [one, dummies], ["group"])}

# 2 x 2 factorial (A0/AY/M0/MY): NaCl, Ni and their interaction
labels = np.array(grp.tolist())
keep = np.isin(labels, ["A0", "AY", "M0", "MY"])
ks = np.where(keep)[0]
Cs = np.eye(len(ks)) - 1 / len(ks)
Gs = Cs @ (-0.5 * dist[np.ix_(ks, ks)] ** 2) @ Cs
g4 = labels[keep]
salt = np.char.endswith(g4, "Y").astype(float)[:, None]
ni = np.char.startswith(g4, "M").astype(float)[:, None]
o4 = np.ones((len(ks), 1))
stats_out["permanova_2x2"] = permanova(Gs, [o4, np.hstack([o4, salt]), np.hstack([o4, salt, ni]),
                                            np.hstack([o4, salt, ni, salt * ni])], ["NaCl", "Ni", "NaCl x Ni"])

# key genera and Shannon: group means, Tukey letters (5 groups), Ni x NaCl interaction on ln values
alpha = pd.read_csv(MG / "平台导出" / "多样性指数结果_nr_Origin.csv").set_index("Sample")
measures = {g: genus_rel.loc[g, samples] for g in ["Clostridium", "Thermoanaerobacterium", "Lactobacillus",
                                                   "Caldifermentibacillus", "Caldibacillus", "Garciella",
                                                   "Neomoorella", "Symbiobacterium", "Schnuerera"]}
measures["Shannon"] = alpha.loc[samples, "shannon"]
for name, s in measures.items():
    data = [s[grp == g].values for g in GROUPS]
    stats_out[name] = {"mean": {g: float(d.mean()) for g, d in zip(GROUPS, data)},
                       "sd": {g: float(d.std(ddof=1)) for g, d in zip(GROUPS, data)},
                       "letters": compact_letters(GROUPS, data),
                       "p_int_Ni": float(twoway_interaction_p(labels, s.values + 1e-3, "M"))}
json.dump(stats_out, open(D / "fig4_stats.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: stats_out[k] for k in ["explained", "permanova_5groups", "permanova_2x2"]}, indent=1))
for name in measures:
    s = stats_out[name]
    print(f"{name:22s}", {g: round(m, 2) for g, m in s["mean"].items()}, s["letters"], f"P_int={s['p_int_Ni']:.4f}")
