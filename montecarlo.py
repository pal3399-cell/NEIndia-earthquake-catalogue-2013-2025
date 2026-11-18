"""
Monte Carlo experiments on magnitude-conversion regression (Section 3.7 of the paper).

Three relations are compared, all fitted to the same synthetic calibration pairs:
  ORTH : the structural general orthogonal regression line (alpha_hat, beta_hat), Eqs. (6)-(10);
         this is the relation that GOR2 applies directly to observed magnitudes
  GOR1 : the GOR1 prediction relation (alpha*, beta*), Eqs. (13)-(14)
  SLR  : ordinary least squares of the target on the observed predictor

Experiments
  aad_table()       Table 5 : average absolute difference between fitted and true relations
  eta_sensitivity() Table 6 / Fig. 6 : ORTH slope under an assumed error-variance ratio
  b_propagation()   Fig. 7 : b-value recovered after converting a synthetic catalogue

Usage: python montecarlo.py outdir
"""
import json
import sys
import numpy as np

RNG = np.random.default_rng(2026)

# published relations (Pallavi et al. 2023) used as the true lines:
# name: (intercept, slope, lower, upper, n, eta, predictor error s.d.)
PAIRS = {
    "Ms(ISC)":  (1.69, 0.680, 4.1, 6.1, 93, 0.6, 0.15),
    "MS(NEIC)": (1.193, 0.771, 4.2, 6.1, 57, 0.6, 0.15),
    "mb(ISC)":  (-1.19, 1.19, 4.8, 6.1, 116, 0.2, 0.25),
    "mb(NEIC)": (-1.391, 1.21, 4.8, 6.1, 106, 0.2, 0.25),
    "ML(IMD)":  (-1.890, 1.31, 5.0, 6.6, 100, 1.0, 0.20),
    "MD(NEIC)": (-0.832, 0.821, 4.2, 6.8, 376, 1.0, 0.22),
}


def gr_sample(n, b, mmin, mmax=None, rng=None):
    rng = RNG if rng is None else rng
    beta = b * np.log(10)
    if mmax is None:
        return mmin + rng.exponential(1 / beta, n)
    u = rng.random(n)
    return mmin - np.log(1 - u * (1 - np.exp(-beta * (mmax - mmin)))) / beta


def fit_all(mx, my, eta):
    """Return the ORTH, GOR1 and SLR relations as (intercept, slope) pairs."""
    sxx, syy = np.var(mx, ddof=1), np.var(my, ddof=1)
    sxy = np.cov(mx, my, ddof=1)[0, 1]
    d = syy - eta * sxx
    beta = (d + np.sqrt(d * d + 4 * eta * sxy * sxy)) / (2 * sxy)          # Eq. (9)
    alpha = my.mean() - beta * mx.mean()                                   # Eq. (11)
    mxhat = (eta * mx + beta * (my - alpha)) / (eta + beta ** 2)          # Eq. (13)
    bstar, astar = np.polyfit(mx, alpha + beta * mxhat, 1)               # Eq. (14)
    bs, as_ = np.polyfit(mx, my, 1)
    return {"ORTH": (alpha, beta), "GOR1": (astar, bstar), "SLR": (as_, bs)}


def make_pairs(n, a, b, lo, hi, sd_x, eta, bval=0.8):
    mx_true = gr_sample(n, bval, lo, hi)
    mx = mx_true + RNG.normal(0, sd_x, n)
    my = a + b * mx_true + RNG.normal(0, np.sqrt(eta) * sd_x, n)
    return mx, my


def aad_table(reps=1000):
    out = {}
    for name, (a, b, lo, hi, n, eta, sd) in PAIRS.items():
        grid = np.linspace(lo, hi, 200)
        w = 10 ** (-0.8 * grid); w /= w.sum()
        upper = grid >= lo + 0.75 * (hi - lo)
        acc = {k: [] for k in ("ORTH", "GOR1", "SLR")}
        for _ in range(reps):
            rel = fit_all(*make_pairs(n, a, b, lo, hi, sd, eta), eta)
            for k, (aa, bb) in rel.items():
                e = np.abs(aa + bb * grid - (a + b * grid))
                acc[k].append(((e * w).sum(), e[upper].mean()))
        out[name] = {k: [round(float(x), 3) for x in np.mean(v, 0)] for k, v in acc.items()}
        out[name]["n"] = n
    return out


def eta_sensitivity(reps=500, etas=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)):
    out = {}
    for e in etas:
        s = [fit_all(*make_pairs(116, -1.19, 1.19, 4.8, 6.1, 0.25, 0.2), e)["ORTH"][1] for _ in range(reps)]
        out[e] = [round(float(np.mean(s)), 3), round(float(np.std(s)), 3), round(100 * (np.mean(s) / 1.19 - 1), 1)]
    return out


def b_ml(m, mc):
    x = m[m >= mc]
    return np.log10(np.e) / (x.mean() - mc)


def b_propagation(reps=500, ncat=5000, a=-1.19, b=1.19, sd=0.25, eta=0.2, mc=4.0):
    res = {k: [] for k in ("ORTH", "GOR1", "SLR")}
    for _ in range(reps):
        mw_true = gr_sample(ncat, 0.80, 2.8)
        mb_obs = (mw_true - a) / b + RNG.normal(0, sd, ncat)
        rel = fit_all(*make_pairs(116, a, b, 4.8, 6.1, sd, eta), eta)
        for k, (aa, bb) in rel.items():
            res[k].append(b_ml(aa + bb * mb_obs, mc))
    return {k: np.array(v) for k, v in res.items()}


def plot_bprop(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    bins = np.linspace(0.55, 1.6, 64)
    style = {"ORTH": ("#1f5fa8", "Orthogonal fit"), "GOR1": ("#d9822b", "GOR1 relation"), "SLR": ("#c0392b", "SLR")}
    for k, (c, lab) in style.items():
        v = res[k]
        ax.hist(v, bins=bins, color=c, alpha=0.55, label=f"{lab}: {v.mean():.2f} $\\pm$ {v.std():.2f}")
    ax.axvline(0.80, color="k", ls="--", lw=1.2, label="True b = 0.80")
    ax.set_xlabel("Recovered b-value")
    ax.set_ylabel("Number of replications")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    RNG = np.random.default_rng(2026)
    eta = eta_sensitivity()
    RNG = np.random.default_rng(2027)
    aad = aad_table()
    RNG = np.random.default_rng(2028)
    bp = b_propagation()
    plot_bprop(bp, f"{outdir}/bprop.png")
    out = {"eta": {str(k): v for k, v in eta.items()}, "aad": aad,
           "bprop": {k: [round(float(v.mean()), 3), round(float(v.std()), 3)] for k, v in bp.items()}}
    with open(f"{outdir}/montecarlo_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
