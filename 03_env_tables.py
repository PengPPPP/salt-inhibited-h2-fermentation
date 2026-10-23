"""Endpoint environmental-factor tables for sequencing correlation analyses.

Outputs:
  环境因子表_宏基因组终点.csv / env_table_jieyizhi.txt                  (relief batch, all 15 metagenome samples)
  环境因子表_宏基因组终点_剔除A0_1.csv / env_table_jieyizhi_noA0_1.txt  (main analysis)
  16S/env_table_16S_Y0-Y4.txt                                         (salt-gradient batch, 67 h)
H2 comes from the gas bottles and chemistry from the sampling bottles of the same replicate number;
correlations between H2 and community data should therefore use group means.
"""
import openpyxl
import pandas as pd

from common import DILUTION, ML_PER_AREA, ROOT, S16, VFA

MG_SAMPLES = [f"{g}-{i}" for g in ["A0", "AY", "H0", "M0", "MY"] for i in (1, 2, 3)]
Y_SAMPLES = [f"Y{g}-{i}" for g in range(5) for i in (1, 2, 3)]
wb = lambda name: openpyxl.load_workbook(ROOT / name, data_only=True)


def first_rows(ws, col, samples, r0=1, r1=80):
    out = {}
    for r in range(r0, r1):
        v = ws.cell(r, col).value
        if v in samples and v not in out:
            out[v] = r
    return out


def assay_endpoint(files, samples, col_last):
    vals = {}
    for p in files:
        ws = wb(p).worksheets[0]
        for s, r in first_rows(ws, 2, samples).items():
            vals[s] = ws.cell(r, col_last).value * 1000 * DILUTION
    return vals


# ---------------------------------------------------------------- relief batch (metagenome samples)
env = {s: {} for s in MG_SAMPLES}
ws = wb("水热炭解抑制产氢.xlsx")["AH对照"]
for s, r in first_rows(ws, 4, MG_SAMPLES, 1, 45).items():
    env[s]["H2_mL"] = ws.cell(r, 10).value * ML_PER_AREA            # 78 h
ws = wb("活性炭氯化镍 产氢.xlsx")["氢气数据"]
for s, r in first_rows(ws, 12, MG_SAMPLES, 1, 50).items():
    env[s]["H2_mL"] = ws.cell(r, 18).value                          # 80 h
for p, r0 in [("水热炭解抑制VFA.xlsx", 137), ("活性炭氯化镍解抑制VFA.xlsx", 133)]:
    ws = wb(p)["Sheet2"]
    for s, r in first_rows(ws, 3, MG_SAMPLES, r0, r0 + 30).items():
        env[s].update({n + "_mgCOD": ws.cell(r, 25 + j).value for j, n in enumerate(VFA)})
for key, files in [("Carb_mgL", ["水热炭解抑制碳水.xlsx", "活性炭氯化镍 碳水.xlsx"]),
                   ("Protein_mgL", ["水热碳解抑制蛋白质.xlsx", "活性炭氯化镍 蛋白质.xlsx"])]:
    for s, v in assay_endpoint(files, MG_SAMPLES, 11).items():
        env[s][key] = v
df = pd.DataFrame(env).T
df["TotalVFA_mgCOD"] = df[[n + "_mgCOD" for n in VFA]].sum(axis=1)
df["Butyrate_pct"] = df.Butyrate_mgCOD / df.TotalVFA_mgCOD * 100
df.insert(0, "Batch", ["Jul" if s[0] == "M" else "Jun" for s in df.index])
df.insert(0, "Hydrochar_gL", [10 if s[0] == "H" else 0 for s in df.index])
df.insert(0, "Ni_mgL", [2 if s[0] == "M" else 0 for s in df.index])
df.insert(0, "NaCl_gL", [20 if s[1] == "Y" else 0 for s in df.index])
df.index = [s.replace("-", "_") for s in df.index]
df.index.name = "Sample"
df = df.round(2)

platform_cols = {"H2_mL": "H2", "Ethanol_mgCOD": "Ethanol", "Propanol_mgCOD": "Propanol", "Acetate_mgCOD": "Acetate",
                 "Propionate_mgCOD": "Propionate", "Butyrate_mgCOD": "Butyrate", "Caproate_mgCOD": "Caproate",
                 "TotalVFA_mgCOD": "TotalVFA", "Carb_mgL": "Carbohydrate", "Protein_mgL": "Protein"}
for suffix, d in [("", df), ("_剔除A0_1", df.drop(index="A0_1"))]:
    d.to_csv(ROOT / f"环境因子表_宏基因组终点{suffix}.csv", encoding="utf-8-sig", lineterminator="\r\n")
    plat = d[list(platform_cols)].rename(columns=platform_cols)
    plat.index.name = "#SampleID"
    name = "env_table_jieyizhi.txt" if not suffix else "env_table_jieyizhi_noA0_1.txt"
    plat.to_csv(ROOT / name, sep="\t", lineterminator="\r\n", float_format="%.2f")

# ---------------------------------------------------------------- salt-gradient batch (16S samples), 67 h
envy = {s: {"NaCl": int(s[1]) * 5} for s in Y_SAMPLES}
ws = wb("盐抑制 产氢气.xlsx")["Sheet2"]
for s, r in first_rows(ws, 12, Y_SAMPLES, 1, 30).items():
    envy[s]["H2"] = ws.cell(r, 19).value
ws = wb("盐抑制 VFA.xlsx")["Sheet2"]
for k, s in enumerate(Y_SAMPLES):
    envy[s].update({n: ws.cell(124 + k, 24 + j).value for j, n in enumerate(VFA)})
for key, f in [("Carbohydrate", "盐抑制 碳水化合物.xlsx"), ("Protein", "盐抑制 蛋白质.xlsx")]:
    for s, v in assay_endpoint([f], Y_SAMPLES, 11).items():
        envy[s][key] = v
dy = pd.DataFrame(envy).T
dy.insert(len(dy.columns) - 2, "TotalVFA", dy[VFA].sum(axis=1))
dy.index = [s.replace("-", "_") for s in dy.index]
dy.index.name = "#SampleID"
dy["NaCl"] = dy.NaCl.astype(int)
dy.to_csv(S16 / "env_table_16S_Y0-Y4.txt", sep="\t", lineterminator="\r\n", float_format="%.2f")
print(df.groupby(df.index.str[:2])[["H2_mL", "Butyrate_pct", "Carb_mgL"]].mean().round(2).to_string())
