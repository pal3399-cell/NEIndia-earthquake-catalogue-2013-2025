"""
Reproducible analysis of the NEIC/USGS ComCat catalogue for Northeast India, 2013-2025.

Steps
1. Homogenise magnitudes to Mwg:
   mb -> Mwg   Eq. (18): Mwg = 1.21 mb - 1.391          (Pallavi et al. 2023)
   ML -> Mwg   Eq. (19): Mwg = 1.31 ML - 1.890          (Pallavi et al. 2023)
   Mw -> Mwg   via seismic moment: log10 M0 = 1.5 Mw + 16.05 (dyn cm, Hanks & Kanamori 1979),
               Mwg = log10 M0 / 1.36 - 12.68           (Das et al. 2019)
2. Decluster with Uhrhammer-type windows (regional parameters of Das & Meneses 2021):
   r = exp(-1.022 + 0.804 M) km, t = exp(2.264 + 0.285 M) days,
   largest event first, windows applied before and after the event, epicentral distance.
3. Completeness and Gutenberg-Richter parameters.
   Converted body-wave magnitudes are discretised in steps of 1.21 x 0.1 = 0.121 Mwg units, so
   0.1-unit bins leave empty bins. All binning therefore uses the native step dM = 0.1 * beta_mb,
   with bin centres on the converted mb ladder.
   Mc  : maximum curvature + 0.2, snapped up to the next ladder level (MAXC)
         cross-checked by b-value stability (MBS; Cao & Gao 2002; Woessner & Wiemer 2005)
   b   : Aki-Utsu maximum likelihood with bin correction dM/2; error Shi & Bolt (1982)
   a   : log10 N(M >= Mc) + b Mc
4. Sensitivity: Mw treatment, in-range conversions only, conversion-parameter uncertainty.

Usage: python catalogue_analysis.py raw.csv outdir
"""
import json
import sys
import numpy as np
import pandas as pd

A_MB, C_MB, SA_MB, SC_MB = 1.21, -1.391, 0.016, 0.390
A_ML, C_ML, SA_ML, SC_ML = 1.31, -1.890, 0.005, 0.25
MB_LO, MB_HI, ML_LO, ML_HI = 4.8, 6.1, 5.0, 6.6
PERIODS = {"2013-2016": (2013, 2016), "2017-2020": (2017, 2020),
           "2021-2025": (2021, 2025), "2013-2025": (2013, 2025)}


def load(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["t_days"] = (df.time - df.time.iloc[0]).dt.total_seconds() / 86400.0
    mt = df.magType.str.lower()
    df["route"] = np.where(mt == "mb", "mb", np.where(mt == "ml", "ml", "mw"))
    return df


def to_mwg(df, a_mb=A_MB, c_mb=C_MB, a_ml=A_ML, c_ml=C_ML, mw_mode="das2019"):
    m = df.mag.values.astype(float)
    out = np.empty_like(m)
    r = df.route.values
    out[r == "mb"] = a_mb * m[r == "mb"] + c_mb
    out[r == "ml"] = a_ml * m[r == "ml"] + c_ml
    if mw_mode == "das2019":
        out[r == "mw"] = (1.5 * m[r == "mw"] + 16.05) / 1.36 - 12.68
    else:  # Mw adopted unchanged
        out[r == "mw"] = m[r == "mw"]
    return out


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def decluster(df, M, rc=(-1.022, 0.804), tc=(2.264, 0.285)):
    n = len(df)
    lat, lon, t = df.latitude.values, df.longitude.values, df.t_days.values
    cid = -np.ones(n, int)
    k = 0
    for i in np.argsort(-M, kind="stable"):
        if cid[i] >= 0:
            continue
        r = np.exp(rc[0] + rc[1] * M[i])
        tw = np.exp(tc[0] + tc[1] * M[i])
        idx = np.where((cid < 0) & (np.abs(t - t[i]) <= tw))[0]
        members = idx[haversine(lat[i], lon[i], lat[idx], lon[idx]) <= r]
        cid[members] = k
        cid[i] = k
        k += 1
    main = np.zeros(n, bool)
    for c in np.unique(cid):
        mem = np.where(cid == c)[0]
        main[mem[np.argmax(M[mem])]] = True
    return cid, main


class Ladder:
    """Bins centred on the converted mb levels, width dM = 0.1 * a_mb."""
    def __init__(self, a_mb=A_MB, c_mb=C_MB):
        self.dm = 0.1 * a_mb
        self.c0 = a_mb * 3.0 + c_mb            # converted value of mb 3.0
    def level(self, x):                         # nearest ladder level
        return self.c0 + np.round((np.asarray(x) - self.c0) / self.dm) * self.dm
    def ceil(self, x):                          # smallest ladder level >= x
        return self.c0 + np.ceil((np.asarray(x) - self.c0) / self.dm - 1e-9) * self.dm


def b_aki(m, mc, dm):
    x = m[m >= mc - dm / 2]
    n = len(x)
    b = np.log10(np.e) / (x.mean() - (mc - dm / 2))
    sb = 2.30 * b ** 2 * np.sqrt(((x - x.mean()) ** 2).sum() / (n * (n - 1)))
    return b, sb, n, np.log10(n) + b * mc


def mc_maxc(m, lad, corr=0.2):
    lev = np.round(lad.level(m), 4)
    vals, cnt = np.unique(lev, return_counts=True)
    return float(lad.ceil(vals[np.argmax(cnt)] + corr))


def mc_mbs(m, lad, window=0.5):
    start = lad.ceil(np.min(m))
    steps = int(round(window / lad.dm))
    for j in range(60):
        mco = start + j * lad.dm
        if (m >= mco - lad.dm / 2).sum() < 50:
            break
        b, sb, _, _ = b_aki(m, mco, lad.dm)
        bs = [b_aki(m, mco + s * lad.dm, lad.dm)[0] for s in range(1, steps + 1)
              if (m >= mco + s * lad.dm - lad.dm / 2).sum() >= 10]
        if bs and abs(np.mean(bs + [b]) - b) <= sb:
            return float(mco)
    return np.nan


def gr_table(df, M, main, lad, rng, nboot=150):
    d = df[main].copy()
    d["Mwg"] = M[main]
    rows = {}
    for name, (y0, y1) in PERIODS.items():
        m = d.Mwg.values[d.time.dt.year.between(y0, y1).values]
        mc = mc_maxc(m, lad)
        boot = [mc_maxc(rng.choice(m, len(m)), lad) for _ in range(nboot)]
        b, sb, n, a = b_aki(m, mc, lad.dm)
        mbs = mc_mbs(m, lad)
        bm, sbm, nm, _ = b_aki(m, mbs, lad.dm)
        rows[name] = dict(events=int(len(m)), Mc=round(mc, 2), Mc_sd=round(float(np.std(boot)), 2),
                          N=int(n), b=round(b, 2), b_sd=round(sb, 2), a=round(a, 2),
                          Mc_MBS=round(mbs, 2), N_MBS=int(nm), b_MBS=round(bm, 2), b_MBS_sd=round(sbm, 2))
    return rows


def plot_fmd(df, M, mainflag, lad, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[mainflag].copy()
    d["Mwg"] = M[mainflag]
    fig, axs = plt.subplots(2, 2, figsize=(9, 7), dpi=200, sharex=True, sharey=True)
    for ax, (name, (y0, y1)), tag in zip(axs.flat, PERIODS.items(), "abcd"):
        m = d.Mwg.values[d.time.dt.year.between(y0, y1).values]
        lev = np.round(lad.level(m), 4)
        vals, cnt = np.unique(lev, return_counts=True)
        cum = np.array([(lev >= v - 1e-9).sum() for v in vals])
        ax.semilogy(vals, cum, "s", ms=4, mfc="none", mec="#1f5fa8", label="Cumulative")
        ax.semilogy(vals, cnt, "^", ms=4, mfc="none", mec="#555555", label="Non-cumulative")
        g = res["gr"][name]
        x = np.linspace(g["Mc"], vals.max(), 50)
        ax.semilogy(x, 10 ** (g["a"] - g["b"] * x), "-", color="#c0392b", lw=1.4,
                    label=f"Fit above $M_c$: b = {g['b']:.2f}")
        ax.axvline(g["Mc"], color="k", ls="--", lw=1)
        title = f"({tag}) {name}: $M_c$ = {g['Mc']:.2f}"
        if name == "2013-2025":
            r = res["inrange"]
            x2 = np.linspace(r["T"], vals.max(), 50)
            ax.semilogy(x2, 10 ** (r["a"] - r["b"] * x2), "-", color="#1e8449", lw=1.6,
                        label=f"Fit above {r['T']:.2f}: b = {r['b']:.2f}")
            ax.axvline(r["T"], color="#1e8449", ls=":", lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.grid(alpha=0.25, which="both", lw=0.4)
    for ax in axs[1]:
        ax.set_xlabel("$M_{wg}$")
    for ax in axs[:, 0]:
        ax.set_ylabel("Number of events")
    fig.tight_layout()
    fig.savefig(path)


def main(path, outdir):
    rng = np.random.default_rng(20261118)
    df = load(path)
    lad = Ladder()
    M = to_mwg(df)
    cid, mainflag = decluster(df, M)
    sizes = pd.Series(cid).value_counts()
    top = sizes.head(3)
    res = dict(n_events=len(df),
               routes=df.route.value_counts().to_dict(),
               mb_below=int(((df.route == "mb") & (df.mag < MB_LO)).sum()),
               mb_in=int(((df.route == "mb") & (df.mag >= MB_LO) & (df.mag <= MB_HI)).sum()),
               ml_below=int(((df.route == "ml") & (df.mag < ML_LO)).sum()),
               mainshocks=int(mainflag.sum()), removed=int((~mainflag).sum()),
               removed_pct=round(100 * (~mainflag).sum() / len(df), 2),
               clusters=int(len(sizes)), multi_clusters=int((sizes > 1).sum()),
               top3=[[int(s), round(float(M[cid == c].max()), 2)] for c, s in top.items()],
               dM=round(lad.dm, 3),
               big=sorted([round(float(x), 2) for x in M[M >= 6.45]], reverse=True))
    res["gr"] = gr_table(df, M, mainflag, lad, rng)
    full = res["gr"]["2013-2025"]

    # sensitivity: events above Mc converted by extrapolation
    d = df[mainflag].copy()
    d["Mwg"] = M[mainflag]
    mc = full["Mc"]
    above = d[d.Mwg >= mc - lad.dm / 2]
    ext = above[((above.route == "mb") & (above.mag < MB_LO)) | ((above.route == "ml") & (above.mag < ML_LO))]
    res["above_Mc"] = int(len(above))
    res["above_Mc_extrapolated"] = int(len(ext))
    res["extrap_mb_values"] = sorted(float(x) for x in ext.mag.unique())
    # in-range only: lowest ladder level that excludes every extrapolated conversion
    t_in = float(lad.ceil(A_MB * MB_LO + C_MB))
    b, sb, n, _ = b_aki(d.Mwg.values, t_in, lad.dm)
    res["inrange"] = dict(T=round(t_in, 2), N=int(n), b=round(b, 2), b_sd=round(sb, 2),
                          extrapolated_in_sample=int((((d.route == "mb") & (d.mag < MB_LO)) &
                                                      (d.Mwg >= t_in - lad.dm / 2)).sum()))
    # threshold scan for both Mw treatments
    scan = {}
    for mode in ("das2019", "identity"):
        Mx = to_mwg(df, mw_mode=mode)
        cx, mx = decluster(df, Mx)
        mm = Mx[mx]
        mcx = mc_maxc(mm, lad)
        scan[mode] = dict(mainshocks=int(mx.sum()), Mc=round(mcx, 2),
                          rows=[[round(T, 2)] + [round(v, 2) if isinstance(v, float) else int(v)
                                                 for v in b_aki(mm, T, lad.dm)[:3]]
                                for T in [mcx + s * lad.dm for s in range(0, 6)]])
    res["scan"] = scan
    # conversion-parameter uncertainty (independent normal; no covariance is published)
    mcs, bs, ns, bi = [], [], [], []
    for _ in range(1000):
        am, cm = rng.normal(A_MB, SA_MB), rng.normal(C_MB, SC_MB)
        al, cl = rng.normal(A_ML, SA_ML), rng.normal(C_ML, SC_ML)
        lx = Ladder(am, cm)
        Mx = to_mwg(df, am, cm, al, cl)
        cx, mx = decluster(df, Mx)
        mm = Mx[mx]
        mcx = mc_maxc(mm, lx)
        bx, _, nx, _ = b_aki(mm, mcx, lx.dm)
        tx = float(lx.ceil(am * MB_LO + cm))
        bi.append(b_aki(mm, tx, lx.dm)[0])
        mcs.append(mcx); bs.append(bx); ns.append(nx)
    res["propagation"] = dict(Mc_med=round(float(np.median(mcs)), 2),
                              Mc_lo=round(float(np.percentile(mcs, 2.5)), 2),
                              Mc_hi=round(float(np.percentile(mcs, 97.5)), 2),
                              b_sd_conv=round(float(np.std(bs)), 3),
                              b_med=round(float(np.median(bs)), 2),
                              N_lo=int(np.percentile(ns, 2.5)), N_hi=int(np.percentile(ns, 97.5)),
                              b_total_sd=round(float(np.hypot(np.std(bs), full["b_sd"])), 2),
                              b_in_med=round(float(np.median(bi)), 2), b_in_sd_conv=round(float(np.std(bi)), 3))
    # sensitivity to the choice of conversion relation (Pallavi et al. 2023, Table 2)
    rel = {"GOR1": (A_MB, C_MB, A_ML, C_ML), "GOR2": (1.68, -3.89, 1.47, -2.76), "SLR": (1.18, -1.27, 1.24, -1.49)}
    res["relations"] = {}
    for name, (am, cm, al, cl) in rel.items():
        lx = Ladder(am, cm)
        Mx = to_mwg(df, am, cm, al, cl)
        cx, mx = decluster(df, Mx)
        mm = Mx[mx]
        mcx = mc_maxc(mm, lx)
        b0, s0, n0, _ = b_aki(mm, mcx, lx.dm)
        tx = float(lx.ceil(am * MB_LO + cm))
        b1, s1, n1, a1 = b_aki(mm, tx, lx.dm)
        res["relations"][name] = dict(Mc=round(mcx, 2), N=int(n0), b=round(b0, 2), b_sd=round(s0, 2),
                                      T=round(tx, 2), N_in=int(n1), b_in=round(b1, 2), b_in_sd=round(s1, 2))
    res["inrange"]["a"] = round(float(np.log10(res["inrange"]["N"]) + res["inrange"]["b"] * res["inrange"]["T"]), 2)
    plot_fmd(df, M, mainflag, lad, res, f"{outdir}/fmd.png")
    # outputs
    out = df[["id", "time", "latitude", "longitude", "depth", "mag", "magType"]].copy()
    out["Mwg"] = np.round(M, 3)
    out["conversion"] = np.select([df.route == "mb", df.route == "ml"],
                                  ["Eq.18 (mb)", "Eq.19 (ML)"], "Das2019 via M0 (Mw)")
    inside = np.select([df.route == "mb", df.route == "ml"],
                       [(df.mag >= MB_LO) & (df.mag <= MB_HI), (df.mag >= ML_LO) & (df.mag <= ML_HI)], False)
    out["calibration_range"] = np.where(df.route == "mw", "not applicable", np.where(inside, "inside", "below"))
    out["cluster_id"] = cid
    out["mainshock"] = mainflag
    out.to_csv(f"{outdir}/NEIndia_2013-2025_Mwg_homogenised.csv", index=False)
    with open(f"{outdir}/results.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=1))
