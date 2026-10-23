"""Salt-gradient batch (Y0–Y4): tidy H2 / liquid-phase data, Gompertz fits and statistics.

Outputs (图表/Fig1/): fig1_h2_timecourse.csv, fig1_chem_timecourse.csv, TableS3_Gompertz_Y.csv, fig1_stats.json
"""
import json

import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

from common import DILUTION, FIG, NACL, ROOT, VFA, VS_SALT, compact_letters, fit_gompertz

OUT = FIG / "Fig1"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES = [f"Y{g}-{i}" for g in range(5) for i in (1, 2, 3)]
H2_TIMES = [0, 15, 24, 36, 46, 58, 67]
SAMPLE_TIMES = [15, 36, 46, 67]           # 1st, 3rd, 4th and 6th sampling


def wb(name):
    return openpyxl.load_workbook(ROOT / name, data_only=True)


# H2: cumulative mL (vented after each reading, increments summed); per-time-point standard gas
ws = wb("盐抑制 产氢气.xlsx")["Sheet2"]
h2 = {}
for r in range(1, 30):
    s = ws.cell(r, 12).value
    if s in SAMPLES and s not in h2:
        h2[s] = [ws.cell(r, 13 + k).value for k in range(len(H2_TIMES))]
h2 = pd.DataFrame(h2, index=H2_TIMES).T

# VFA / alcohols (mg COD/L): blocks of 15 samples per sampling time
ws = wb("盐抑制 VFA.xlsx")["Sheet2"]
rows = []
for b, t in enumerate(SAMPLE_TIMES):
    for k, s in enumerate(SAMPLES):
        rows.append({"Sample": s, "Time": t, **{n: ws.cell(79 + 15 * b + k, 24 + j).value for j, n in enumerate(VFA)}})
vfa = pd.DataFrame(rows)
vfa["TotalVFA"] = vfa[VFA].sum(axis=1)


def assay(name, col):
    ws = wb(name).worksheets[0]
    out, seen = [], set()
    for r in range(1, 80):
        s = ws.cell(r, 2).value
        if s in SAMPLES and s not in seen:
            seen.add(s)
            out += [{"Sample": s, "Time": t, col: ws.cell(r, 8 + k).value * 1000 * DILUTION}
                    for k, t in enumerate(SAMPLE_TIMES)]
    return pd.DataFrame(out)


chem = vfa.merge(assay("盐抑制 碳水化合物.xlsx", "Carbohydrate"), on=["Sample", "Time"]) \
          .merge(assay("盐抑制 蛋白质.xlsx", "Protein"), on=["Sample", "Time"])
chem["Group"] = chem.Sample.str[:2]
chem["NaCl"] = chem.Group.map(NACL)

# Gompertz fits per bottle
fits = []
for s, y in h2.iterrows():
    (P, Rm, lam), r2 = fit_gompertz(H2_TIMES, y.values)
    fits.append({"Sample": s, "Group": s[:2], "NaCl": NACL[s[:2]], "Hmax_mL": P, "Rm_mL_h": Rm, "lambda_h": lam,
                 "R2": r2, "H2_67h_mL": y[67], "Yield_mL_gVS": y[67] / VS_SALT})
fits = pd.DataFrame(fits)

# statistics: one-way ANOVA + Tukey letters, linear trend against NaCl
end = chem[chem.Time == 67].copy()
end["Butyrate_pct"] = end.Butyrate / end.TotalVFA * 100
summary = {}
for name, df, col in [("H2_67h", fits, "H2_67h_mL"), ("Hmax", fits, "Hmax_mL"), ("Rm", fits, "Rm_mL_h"),
                      ("lambda", fits, "lambda_h"), ("Yield", fits, "Yield_mL_gVS"),
                      ("TotalVFA_67h", end, "TotalVFA"), ("Butyrate_67h", end, "Butyrate"),
                      ("Butyrate_pct_67h", end, "Butyrate_pct"), ("Acetate_67h", end, "Acetate"),
                      ("Propionate_67h", end, "Propionate"), ("Ethanol_67h", end, "Ethanol"),
                      ("Carb_67h", end, "Carbohydrate"), ("Protein_67h", end, "Protein")]:
    groups = sorted(df.Group.unique())
    data = [df[df.Group == g][col].values for g in groups]
    reg = stats.linregress(df.NaCl, df[col])
    summary[name] = {"mean": df.groupby("Group")[col].mean().round(3).to_dict(),
                     "sd": df.groupby("Group")[col].std().round(3).to_dict(),
                     "letters": compact_letters(groups, data), "p_anova": stats.f_oneway(*data).pvalue,
                     "slope_per_gL": reg.slope, "r_linear": reg.rvalue, "p_linear": reg.pvalue}
base = summary["H2_67h"]["mean"]["Y0"]
summary["inhibition_pct"] = {g: round((1 - v / base) * 100, 1) for g, v in summary["H2_67h"]["mean"].items()}

h2.to_csv(OUT / "fig1_h2_timecourse.csv", encoding="utf-8-sig")
chem.to_csv(OUT / "fig1_chem_timecourse.csv", index=False, encoding="utf-8-sig")
fits.round(4).to_csv(OUT / "TableS3_Gompertz_Y.csv", index=False, encoding="utf-8-sig")
json.dump(summary, open(OUT / "fig1_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(fits.groupby("Group")[["H2_67h_mL", "Hmax_mL", "Rm_mL_h", "lambda_h"]].mean().round(2).to_string())
print("inhibition (%)", summary["inhibition_pct"])
