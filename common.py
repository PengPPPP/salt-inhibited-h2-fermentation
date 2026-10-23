"""Shared paths, colours, plotting style and statistics helpers."""
import itertools
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]      # project folder (raw Excel files live here)
FIG = ROOT / "图表"
MG = ROOT / "宏基因组"
S16 = ROOT / "16S"

# salt-gradient batch (16S)
SALT_GROUPS = ["Y0", "Y1", "Y2", "Y3", "Y4"]
NACL = dict(zip(SALT_GROUPS, [0, 5, 10, 15, 20]))
RAMP = dict(zip(SALT_GROUPS, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]))   # ordinal blue ramp

# relief batch (metagenome); HY shares the hydrochar hue in a darker step
GROUP_COL = {"A0": "#2a78d6", "AY": "#eb6834", "H0": "#1baf7a", "HY": "#0f7a55", "M0": "#eda100", "MY": "#e87ba4"}
GROUP_LABEL = {"A0": "Control", "AY": "NaCl", "H0": "HC", "HY": "HC + NaCl", "M0": "Ni", "MY": "Ni + NaCl"}

VFA = ["Ethanol", "Propanol", "Acetate", "Propionate", "Butyrate", "Caproate"]
VFA_ORDER = ["Butyrate", "Acetate", "Propionate", "Ethanol", "Propanol", "Caproate"]
VFA_COL = dict(zip(VFA_ORDER, ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
INK, MUTED = "#0b0b0b", "#898781"

DILUTION = 4                               # carbohydrate / protein assays
VS_SLURRY = 77.05 / 6                      # g VS/L after 6x dilution
VS_SALT = VS_SLURRY * 0.040                # 40 mL slurry per bottle
VS_RELIEF = VS_SLURRY * 0.030              # 30 mL slurry per bottle
ML_PER_AREA = 17.8240414019715 / 111461.3  # relief batch: mL H2 per GC peak-area unit (workbook calibration)


def apply_style():
    """Times New Roman bold, boxed axes, outward ticks (group house style)."""
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.weight": "bold", "font.size": 7.5,
        "axes.labelsize": 8, "axes.labelweight": "bold", "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.8, "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman:bold",
        "mathtext.it": "Times New Roman:italic:bold", "mathtext.bf": "Times New Roman:bold",
        "axes.edgecolor": INK, "axes.linewidth": 0.9, "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9, "axes.labelcolor": INK, "savefig.dpi": 600})


def tag(ax, letter, x=-0.26):
    ax.text(x, 1.03, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def pfmt(p):
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"


def save(fig, path_stem, tif=False):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf") + (("tif",) if tif else ()):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(f"{path_stem}.{ext}", **kw)


# ---------------------------------------------------------------- models and statistics
def gompertz(t, P, Rm, lam):
    """Modified Gompertz model (Zwietering et al., 1990)."""
    return P * np.exp(-np.exp(Rm * np.e / P * (lam - t) + 1))


def fit_gompertz(t, y, truncate_at_max=False):
    t, y = np.asarray(t, float), np.asarray(y, float)
    if truncate_at_max:                     # exclude later H2 consumption
        i = int(np.argmax(y)); t, y = t[: i + 1], y[: i + 1]
    p0 = [y.max() * 1.05, max(np.diff(y) / np.diff(t)), 5.0]
    popt, _ = curve_fit(gompertz, t, y, p0=p0, bounds=([0, 0, 0], [200, 20, 80]), maxfev=50000)
    pred = gompertz(t, *popt)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return popt, r2


def compact_letters(groups, data, alpha=0.05):
    """Tukey HSD compact letter display; letters assigned from the highest mean downwards."""
    res = stats.tukey_hsd(*data)
    means = {g: np.mean(d) for g, d in zip(groups, data)}
    order = sorted(groups, key=lambda g: -means[g])
    same = lambda a, b: res.pvalue[groups.index(a), groups.index(b)] >= alpha
    subsets = [set(c) for n in range(len(order), 0, -1) for c in itertools.combinations(order, n)
               if all(same(a, b) for a in c for b in c if a != b)]
    cliques = [c for c in subsets if not any(c < o for o in subsets)]
    cliques.sort(key=lambda c: min(order.index(g) for g in c))
    return {g: "".join(chr(97 + i) for i, c in enumerate(cliques) if g in c) for g in groups}


def twoway_interaction_p(groups, values, additive):
    """2 x 2 ANOVA on ln(values): additive (A vs additive) x NaCl (0 vs 20 g/L); returns P of the interaction."""
    groups, y = np.asarray(groups), np.log(np.asarray(values, float))
    keep = np.isin(groups, ["A0", "AY", additive + "0", additive + "Y"])
    g, y = groups[keep], y[keep]
    salt = np.char.endswith(g.astype(str), "Y").astype(float)
    add = (~np.char.startswith(g.astype(str), "A")).astype(float)
    X = np.column_stack([np.ones_like(y), salt, add, salt * add])
    rss = lambda M: ((y - M @ np.linalg.lstsq(M, y, rcond=None)[0]) ** 2).sum()
    full, dfe = rss(X), len(y) - 4
    return 1 - stats.f.cdf((rss(X[:, :3]) - full) / (full / dfe), 1, dfe)


def quadratic_test(x, y):
    """Coefficients of y ~ x + x^2 and the P value of the quadratic term (extra-sum-of-squares F test)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X1 = np.column_stack([np.ones_like(x), x]); X2 = np.column_stack([X1, x ** 2])
    b1 = np.linalg.lstsq(X1, y, rcond=None)[0]; b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    rss1, rss2 = ((y - X1 @ b1) ** 2).sum(), ((y - X2 @ b2) ** 2).sum()
    F = (rss1 - rss2) / (rss2 / (len(y) - 3))
    return b2, 1 - stats.f.cdf(F, 1, len(y) - 3)


def spearman_exact(x, y):
    """Spearman rho with an exact two-sided permutation P (for very small n)."""
    rho = stats.spearmanr(x, y).statistic
    ry = stats.rankdata(y)
    perms = [stats.spearmanr(stats.rankdata(x), p).statistic for p in itertools.permutations(ry)]
    return rho, float(np.mean([abs(r) >= abs(rho) - 1e-12 for r in perms]))
