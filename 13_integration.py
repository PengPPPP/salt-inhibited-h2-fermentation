"""Integration of community, functional genes and fermentation products (relief batch, metagenome samples, A0_1
excluded): distance-based RDA of the species community constrained by the liquid products, and Spearman
correlations of key genera with products and key genes. H2 comes from separate gas bottles, so it enters only as
group means fitted passively onto the ordination.

Inputs: 宏基因组/物种与通路功能数据库/NR物种注释/<sample>/tax_s.xls, env_table_jieyizhi_noA0_1.txt,
        图表/Fig4/fig4_genus_per_sample.csv, 图表/Fig5/fig5_genes_cpm_per_sample.csv
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from common import FIG, MG, ROOT

D = FIG / "Fig6"
D.mkdir(parents=True, exist_ok=True)
GROUPS = ["A0", "AY", "H0", "M0", "MY"]
N_PERM = 9999
rng = np.random.default_rng(2026)

env = pd.read_csv(ROOT / "env_table_jieyizhi_noA0_1.txt", sep="\t", index_col=0)
samples = list(env.index)
grp = np.array([s[:2] for s in samples])

cols = {}
for f in sorted((MG / "物种与通路功能数据库" / "NR物种注释").glob("*/tax_s.xls")):
    t = pd.read_csv(f, sep="\t")
    cols[t.columns[1]] = t.set_index("#Taxonomy")[t.columns[1]]
sp = pd.DataFrame(cols).fillna(0)[samples]
sp = (sp / sp.sum()).T.values
dist = np.array([[np.abs(a - b).sum() / (a + b).sum() for b in sp] for a in sp])
n = len(samples)
C = np.eye(n) - 1 / n
G = C @ (-0.5 * dist ** 2) @ C


# ---------------------------------------------------------------- constraints: liquid products, VIF < 10
def vif(X):
    R = np.corrcoef(X, rowvar=False)
    return np.diag(np.linalg.inv(R))


cand = ["Butyrate", "Acetate", "Ethanol", "Propionate", "Propanol", "Caproate", "Carbohydrate", "Protein"]
keep = list(cand)
while True:
    v = vif(env[keep].values)
    if v.max() < 10:
        break
    keep.pop(int(np.argmax(v)))
Z = stats.zscore(env[keep].values, ddof=1)


def hat(X):
    Xc = np.column_stack([np.ones(len(X)), X])
    return Xc @ np.linalg.pinv(Xc) - 1 / len(X)          # projection onto the centred constraints


def fstat(Gp, X):
    H = hat(X)
    ss_c, ss_t = np.trace(H @ Gp @ H), np.trace(Gp)
    q = X.shape[1]
    return (ss_c / q) / ((ss_t - ss_c) / (n - q - 1)), ss_c


F0, ss_c = fstat(G, Z)
hits = sum(fstat(G[np.ix_(p, p)], Z)[0] >= F0 - 1e-12 for p in (rng.permutation(n) for _ in range(N_PERM)))
res = {"constraints": keep, "vif": dict(zip(keep, map(float, vif(env[keep].values)))),
       "R2": float(ss_c / np.trace(G)), "R2_adj": float(1 - (1 - ss_c / np.trace(G)) * (n - 1) / (n - len(keep) - 1)),
       "F": float(F0), "P": float((hits + 1) / (N_PERM + 1)), "margin": {}}

# marginal test of each term given all others (residuals permuted under the reduced model)
for j, name in enumerate(keep):
    others = np.delete(Z, j, axis=1)
    H_full, H_red = hat(Z), hat(others)
    def f_marg(Gp):
        ss_full, ss_red = np.trace(H_full @ Gp @ H_full), np.trace(H_red @ Gp @ H_red)
        return (ss_full - ss_red) / ((np.trace(Gp) - ss_full) / (n - Z.shape[1] - 1)), ss_full - ss_red
    Fm, ssm = f_marg(G)
    hits = sum(f_marg(G[np.ix_(p, p)])[0] >= Fm - 1e-12 for p in (rng.permutation(n) for _ in range(N_PERM)))
    res["margin"][name] = {"R2": float(ssm / np.trace(G)), "F": float(Fm), "P": float((hits + 1) / (N_PERM + 1))}

# ordination: constrained axes from the eigen-decomposition of HGH; site scores = weighted averages (G u / sqrt(l))
H = hat(Z)
w, u = np.linalg.eigh(H @ G @ H)
w, u = w[::-1][:2], u[:, ::-1][:, :2]
lc = u * np.sqrt(w)
wa = G @ u / np.sqrt(w)
res["axis_R2"] = (w / np.trace(G)).tolist()
biplot = pd.DataFrame([[np.corrcoef(Z[:, j], lc[:, k])[0, 1] for k in range(2)] for j in range(len(keep))],
                      index=keep, columns=["dbRDA1", "dbRDA2"])

# H2 (separate gas bottles) as group means, fitted passively onto the site scores
h2_mean = json.load(open(FIG / "Fig3" / "fig3_stats.json", encoding="utf-8"))["H2_final_mL"]["mean"]
y =stats.zscore(np.array([h2_mean[g] for g in grp]), ddof=1)
beta = np.linalg.lstsq(np.column_stack([np.ones(n), wa]), y, rcond=None)[0][1:]
fit = np.column_stack([np.ones(n), wa]) @ np.r_[0, beta]
res["H2_passive"] = {"group_means": {g: float(h2_mean[g]) for g in GROUPS},
                     "direction": (beta / np.linalg.norm(beta)).tolist(), "r2": float(np.corrcoef(fit, y)[0, 1] ** 2)}

scores = pd.DataFrame({"Sample": samples, "Group": grp, "dbRDA1": wa[:, 0], "dbRDA2": wa[:, 1]})
scores.to_csv(D / "fig6_dbrda_sites.csv", index=False, encoding="utf-8-sig")
biplot.to_csv(D / "fig6_dbrda_biplot.csv", encoding="utf-8-sig")

# ---------------------------------------------------------------- Spearman: genera vs products and key genes
genus = pd.read_csv(FIG / "Fig4" / "fig4_genus_per_sample.csv", index_col=0, encoding="utf-8-sig")[samples]
genes = pd.read_csv(FIG / "Fig5" / "fig5_genes_cpm_per_sample.csv", index_col=0, encoding="utf-8-sig")[samples]
GENERA = ["Clostridium", "Schnuerera", "Thermoanaerobacterium", "Heyndrickxia", "Lactobacillus",
          "Caldifermentibacillus", "Caldibacillus", "Garciella", "Neomoorella", "Symbiobacterium"]
PRODUCTS = ["Butyrate", "Acetate", "Ethanol", "Propionate", "Carbohydrate", "Protein"]
GENES = ["etfAB", "atoDA", "ptb", "buk", "hndD", "ldhA", "nhaC", "mrpA-G", "proVWX", "cbiMNOQ"]
cols = pd.concat([env[PRODUCTS].T, genes.loc[GENES]])
rho = pd.DataFrame(index=GENERA, columns=cols.index, dtype=float)
pv = rho.copy()
for g in GENERA:
    for c in cols.index:
        r = stats.spearmanr(genus.loc[g].values, cols.loc[c].values)
        rho.loc[g, c], pv.loc[g, c] = r.statistic, r.pvalue
q = stats.false_discovery_control(pv.values.ravel()).reshape(pv.shape)
rho.to_csv(D / "fig6_spearman_rho.csv", encoding="utf-8-sig")
pd.DataFrame(q, index=GENERA, columns=cols.index).to_csv(D / "fig6_spearman_q.csv", encoding="utf-8-sig")
res["spearman_n"] = n
json.dump(res, open(D / "fig6_stats.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in res.items()}, indent=1))
print(biplot.round(2))
print(rho.round(2).to_string())
print(pd.DataFrame(q, index=GENERA, columns=cols.index).round(3).to_string())
