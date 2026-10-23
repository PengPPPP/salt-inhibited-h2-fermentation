"""Relief batch (20 g/L NaCl ± hydrochar / Ni): tidy H2 data, endpoint chemistry, Gompertz fits and statistics.

The control replicate pair A0-1 (gas bottle) / A0_1 (sampling bottle) is excluded from the main analysis;
08_figS1_abnormal_replicate.py reports the results with all samples kept.
Outputs (图表/Fig3/): fig3_h2_timecourse.csv, fig3_endpoint_chem.csv, TableS4_Gompertz_relief.csv, fig3_stats.json
"""
import json

import numpy as np
import openpyxl
import pandas as pd

from common import DILUTION, FIG, ML_PER_AREA, ROOT, VFA, VS_RELIEF, fit_gompertz, twoway_interaction_p

OUT = FIG / "Fig3"
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = ["A0", "AY", "H0", "HY", "M0", "MY"]
EXCLUDE = {"A0-1"}


def wb(name):
    return openpyxl.load_workbook(ROOT / name, data_only=True)


# H2 (cumulative headspace mL; bottles were not vented)
rows = []
ws = wb("水热炭解抑制产氢.xlsx")["AH对照"]                     # raw GC peak areas, 19-78 h
for r in range(1, 45):
    s = ws.cell(r, 4).value
    if isinstance(s, str) and s[:2] in ("A0", "AY", "H0", "HY") and s not in EXCLUDE:
        rows += [{"Sample": s, "Time": t, "H2": ws.cell(r, 5 + k).value * ML_PER_AREA}
                 for k, t in enumerate([19, 31, 42, 53, 64, 78])] + [{"Sample": s, "Time": 0, "H2": 0.0}]
ws = wb("活性炭氯化镍 产氢.xlsx")["氢气数据"]                  # calibrated mL, 20-80 h
seen = set()
for r in range(1, 50):
    s = ws.cell(r, 12).value
    if isinstance(s, str) and s[:2] in ("M0", "MY") and s not in seen:
        seen.add(s)
        rows += [{"Sample": s, "Time": t, "H2": ws.cell(r, 13 + k).value}
                 for k, t in enumerate([20, 35, 46, 58, 69, 80])] + [{"Sample": s, "Time": 0, "H2": 0.0}]
h2 = pd.DataFrame(rows)
h2["Group"] = h2.Sample.str[:2]
h2 = h2.sort_values(["Group", "Sample", "Time"])

fits = []
for s, d in h2.groupby("Sample"):
    t, y = d.Time.values, d.H2.values
    (P, Rm, lam), r2 = fit_gompertz(t, y, truncate_at_max=True)
    fits.append({"Sample": s, "Group": s[:2], "Hmax_mL": P, "Rm_mL_h": Rm, "lambda_h": lam, "R2": r2,
                 "H2_final_mL": y[-1], "H2_peak_mL": y.max(), "t_peak_h": t[int(np.argmax(y))],
                 "Yield_final_mL_gVS": y[-1] / VS_RELIEF, "Yield_peak_mL_gVS": y.max() / VS_RELIEF})
fits = pd.DataFrame(fits)

# endpoint chemistry: last sampling of the sampling bottles
chem = []
for p, r0 in [("水热炭解抑制VFA.xlsx", 137), ("活性炭氯化镍解抑制VFA.xlsx", 133)]:
    ws = wb(p)["Sheet2"]
    for r in range(r0, r0 + 30):
        s = ws.cell(r, 3).value
        if isinstance(s, str) and s[:2] in GROUPS and s not in EXCLUDE:
            chem.append({"Sample": s, "Group": s[:2], **{n: ws.cell(r, 25 + j).value for j, n in enumerate(VFA)}})
chem = pd.DataFrame(chem).drop_duplicates("Sample").set_index("Sample")
for p in ["水热炭解抑制碳水.xlsx", "活性炭氯化镍 碳水.xlsx", "水热碳解抑制蛋白质.xlsx", "活性炭氯化镍 蛋白质.xlsx"]:
    key = "Carbohydrate" if "碳水" in p else "Protein"
    ws = wb(p).worksheets[0]
    for r in range(1, 80):
        s = ws.cell(r, 2).value
        if s in chem.index and pd.isna(chem.loc[s].get(key, np.nan)):
            chem.loc[s, key] = ws.cell(r, 11).value * 1000 * DILUTION
chem = chem.reset_index()
chem["TotalVFA"] = chem[VFA].sum(axis=1)
chem["Butyrate_pct"] = chem.Butyrate / chem.TotalVFA * 100

summary = {"n": fits.groupby("Group").size().to_dict()}
for col in ["H2_final_mL", "H2_peak_mL", "Yield_final_mL_gVS"]:
    summary[col] = {"mean": fits.groupby("Group")[col].mean().round(3).to_dict(),
                    "sd": fits.groupby("Group")[col].std().round(3).to_dict()}
for col in ["TotalVFA", "Butyrate_pct", "Butyrate", "Acetate", "Carbohydrate", "Protein"]:
    summary[col] = {"mean": chem.groupby("Group")[col].mean().round(2).to_dict(),
                    "sd": chem.groupby("Group")[col].std().round(2).to_dict()}
for add in ["H", "M"]:
    for col in ["H2_final_mL", "H2_peak_mL"]:
        summary[f"twoway_{add}_{col}"] = {"p_interaction": twoway_interaction_p(fits.Group, fits[col], add)}
for col in ["H2_final_mL", "H2_peak_mL"]:
    summary[f"retention_{col}"] = {}
    for a in ["A", "H", "M"]:
        r = (fits[fits.Group == a + "Y"][col] / fits[fits.Group == a + "0"][col].mean()).values
        summary[f"retention_{col}"][a] = {"mean": float(r.mean()), "sd": float(r.std(ddof=1)), "values": r.round(3).tolist()}

h2.to_csv(OUT / "fig3_h2_timecourse.csv", index=False, encoding="utf-8-sig")
chem.round(2).to_csv(OUT / "fig3_endpoint_chem.csv", index=False, encoding="utf-8-sig")
fits.round(4).to_csv(OUT / "TableS4_Gompertz_relief.csv", index=False, encoding="utf-8-sig")
json.dump(summary, open(OUT / "fig3_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(fits.groupby("Group")[["H2_final_mL", "H2_peak_mL"]].mean().round(2).to_string())
print({k: round(v["p_interaction"], 4) for k, v in summary.items() if k.startswith("twoway")})
print({a: round(v["mean"], 2) for a, v in summary["retention_H2_final_mL"].items()})
