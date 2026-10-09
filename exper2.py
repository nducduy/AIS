import sys, json, time, numpy as np
from scipy import stats, optimize, special
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ais2 import *
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": .3})
STAGE = sys.argv[1]; ARG = sys.argv[2] if len(sys.argv) > 2 else ""; R_ = {}; FIG = "figs/"
std_logp = lambda d: (lambda z: -0.5 * np.sum(z ** 2, 1) - 0.5 * d * LOG2PI)

def run_mean(meth, tg, d, n, B, seed, **kw):
    rng = np.random.default_rng(seed)
    if meth == "SN-AIS":
        return sn_mean(tg, d, n, B, rng, **kw)
    m, g0 = meth.split(":") if ":" in meth else (meth, "1")
    return baseline(m, tg, d, n, B, rng, gamma0=float(g0), **kw)

# ===================================================== E1 exponential integrand
if STAGE == "E1":
    d = 10; a = np.full(d, 2 / np.sqrt(d)); logp = std_logp(d)
    V = lambda nu: np.exp(0.5 * nu @ nu + 0.5 * (2 * a - nu) @ (2 * a - nu)) if np.all(np.isfinite(nu)) else np.inf
    Vs = np.exp(a @ a); n, B, reps = 20000, 100, 20
    meths = ["SN-AIS", "nsgd:0.3", "oais:0.1", "oais:0.3", "oais:1", "ce", "newton"]
    final = {}; curves = {}
    for sc in [1.0, 1e3]:
        for m in meths:
            if sc != 1.0 and m not in ("SN-AIS", "oais:0.3", "nsgd:0.3"):
                continue
            tr = []
            for r in range(reps):
                out = run_mean(m, Target(logp, logf2=lambda z: 2 * z @ a + 2 * np.log(sc)), d, n, B, r)
                tr.append([V(nu) / Vs - 1 for nu in out["nus"]])
            tr = np.array(tr)
            final[f"{m}|{sc}"] = {"median": float(np.median(tr[:, -1])), "q90": float(np.quantile(tr[:, -1], .9)),
                                  "div": int(np.sum(~(tr[:, -1] < 1e3)))}
            if sc == 1.0:
                curves[m] = np.median(tr, 0).tolist()
            print(sc, m, final[f"{m}|{sc}"], flush=True)
    R_["E1"] = {"final": final, "curves": curves, "B": B}
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); ns = np.arange(len(curves["SN-AIS"])) * B
    lab = {"SN-AIS": "SN-AIS", "nsgd:0.3": "normalised SGD ($\\gamma_0=0.3$)", "oais:0.3": "OAIS-type SGD ($\\gamma_0=0.3$)",
           "ce": "CE", "newton": "batch Newton"}
    for m, l in lab.items():
        c = np.array(curves[m]); c[~np.isfinite(c)] = np.nan; ax.loglog(ns[1:], c[1:], label=l)
    ax.set_xlabel("number of samples $n$"); ax.set_ylabel(r"$V(\nu_n)/V(\nu^*)-1$ (median)"); ax.set_ylim(1e-5, 1e2)
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG + "e1.pdf"); plt.close(fig)

# ===================================================== E2 rare events (ARG = c)
if STAGE == "E2":
    d = 10; u = np.ones(d) / np.sqrt(d); logp = std_logp(d); c = float(ARG)
    n, B, reps = 50000, 100, 30; P = stats.norm.sf(c)
    V = lambda nu: np.exp(nu @ nu + stats.norm.logsf(c + u @ nu)) if np.all(np.isfinite(nu)) else np.inf
    ms = optimize.minimize_scalar(lambda m: m * m + stats.norm.logsf(c + m), bounds=(0, 2 * c + 2), method="bounded").x
    lf2 = lambda z: np.where(z @ u > c, 0.0, -np.inf)
    row = {"P": P, "mstar": ms, "rv_opt": V(ms * u) / P ** 2 - 1, "rv_mc": 1 / P - 1, "lam_c": float(stats.norm.pdf(c) / P)}
    g0 = 0.3   # OAIS step tuned in E1 (normalised by first-batch estimate, so scale-free)
    for m in ["SN-AIS", f"oais:{g0}", "ce", "newton"]:
        rv = []; err = []
        for r in range(reps):
            out = run_mean(m, Target(logp, logf2=lf2), d, n, B, 100 + r)
            rv.append(V(out["nus"][-1]) / P ** 2 - 1); err.append(ais_estimate(out["batches"], logf2=lf2) / P - 1)
        row[m] = {"rv_median": float(np.median(rv)), "rv_q90": float(np.quantile(rv, .9)), "rrmse": float(np.sqrt(np.mean(np.square(err))))}
        print(c, m, row[m], flush=True)
    R_["E2"] = {c: row}

# ===================================================== E3 Asian option (ARG = K)
def asian(d, S0=100.0, sig=0.3, r=0.05, T=1.0, K=100.0):
    dt = T / d; drift = (r - 0.5 * sig ** 2) * dt
    def payoff(z):
        logS = np.log(S0) + np.cumsum(drift + sig * np.sqrt(dt) * z, 1)
        return np.exp(-r * T) * np.maximum(np.exp(logS).mean(1) - K, 0.0)
    return payoff

if STAGE == "E3":
    d = 64; logp = std_logp(d); K = float(ARG); pay = asian(d, K=K)
    lf2 = lambda z: np.where((pp := pay(z)) > 0, 2 * np.log(np.maximum(pp, 1e-300)), -np.inf)
    fz = pay(np.random.default_rng(7).standard_normal((200000, d))); price = fz.mean(); var_mc = fz.var()
    ghs = optimize.minimize(lambda z: -(np.log(max(pay(z[None])[0], 1e-300)) - 0.5 * z @ z), np.full(d, 0.3), method="L-BFGS-B").x
    def isv(nu, M=50000, seed=11):
        z = nu + np.random.default_rng(seed).standard_normal((M, d)); fw = pay(z) * np.exp(logp(z) - logq_iso(z, nu))
        return fw.var(), fw.mean()
    vg, mg = isv(ghs, 200000)
    row = {"price": price, "vrf_ghs": var_mc / vg}
    for m in ["SN-AIS", "newton", "oais:0.3", "ce"]:
        vrf = []; tm = []; pe = []
        for r in range(10):
            t0 = time.perf_counter(); out = run_mean(m, Target(logp, logf2=lf2), d, 20000, 200, 200 + r)
            tm.append(time.perf_counter() - t0); vv, mm = isv(out["nus"][-1], seed=300 + r)
            vrf.append(var_mc / vv); pe.append(abs(mm / price - 1))
        row[m] = {"vrf": float(np.median(vrf)), "time": float(np.median(tm)), "price_err": float(max(pe))}
        print(K, m, row[m], flush=True)
    R_["E3"] = {K: row}

# ===================================================== E5 covariance adaptation (closed form)
if STAGE == "E5":
    d = 10; lam = np.geomspace(0.3, 1.2, d); a2 = np.full(d, 0.6)
    logp2 = lambda z: -0.5 * np.sum(z ** 2 / lam, 1) - 0.5 * d * LOG2PI - 0.5 * np.log(lam).sum()
    def logVd(mu, s2):
        A = 2 / lam - 1 / s2
        if np.any(A <= 0) or not np.all(np.isfinite(mu)):
            return np.inf
        b = 2 * a2 - mu / s2
        return np.sum(0.5 * b * b / A - 0.5 * np.log(A) - np.log(lam) + 0.5 * mu ** 2 / s2 + 0.5 * np.log(s2))
    lVs = logVd(lam * a2, lam); mo = optimize.minimize(lambda m: logVd(m, np.ones(d)), np.zeros(d)).fun
    n, B, reps = 20000, 100, 20; curves = {}; final = {}
    def ce_diag(seed):
        rng = np.random.default_rng(seed); mu = np.zeros(d); s2 = np.ones(d); hist = [(mu, s2)]
        for t in range(n // B):
            z = mu + np.sqrt(s2) * rng.standard_normal((B, d)); lw = logp2(z) - logq_diag(z, mu, s2) + z @ a2
            w = np.exp(lw - lw.max()); w /= w.sum()
            mu = w @ z; s2 = np.maximum(w @ (z - mu) ** 2, 1e-3); hist.append((mu.copy(), s2.copy()))
        return hist
    for lab in ["SN-AIS (mean + diag. cov.)", "SN-AIS (mean only)", "CE (mean + diag. cov.)"]:
        tr = []
        for r in range(reps):
            if lab.startswith("SN-AIS (mean +"):
                h = sn_diag(Target(logp2, logf2=lambda z: 2 * z @ a2), d, n, B, np.random.default_rng(r))["hist"]
            elif lab.startswith("SN-AIS (mean only)"):
                out = sn_mean(Target(logp2, logf2=lambda z: 2 * z @ a2), d, n, B, np.random.default_rng(r))
                h = [(nu, np.ones(d)) for nu in out["nus"]]
            else:
                h = ce_diag(r)
            tr.append([np.exp(logVd(*x) - lVs) - 1 for x in h])
        tr = np.array(tr); curves[lab] = np.median(tr, 0).tolist()
        final[lab] = {"median": float(np.median(tr[:, -1])), "q90": float(np.quantile(tr[:, -1], .9)), "div": int(np.sum(~(tr[:, -1] < 1e3)))}
        print(lab, final[lab], flush=True)
    R_["E5"] = {"final": final, "start": float(np.exp(logVd(np.zeros(d), np.ones(d)) - lVs) - 1), "best_mean_only": float(np.exp(mo - lVs) - 1)}
    fig, ax = plt.subplots(figsize=(5.2, 3.4)); ns = np.arange(len(curves["CE (mean + diag. cov.)"])) * B
    for lab, c in curves.items():
        c = np.array(c); ax.loglog(np.arange(len(c))[1:] * B, c[1:], label=lab)
    ax.axhline(np.exp(mo - lVs) - 1, color="k", ls=":", lw=.8, label="best mean-only proposal")
    ax.set_xlabel("number of samples $n$"); ax.set_ylabel(r"$V/V^*-1$ (median)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG + "e5.pdf"); plt.close(fig)

# ===================================================== E4 / E6 Bayesian logistic regression (ARG = synthetic | cancer)
if STAGE == "E4":
    if ARG == "synthetic":
        rng = np.random.default_rng(3); nobs, p = 40, 5
        X = np.column_stack([np.ones(nobs), rng.standard_normal((nobs, p - 1))])
        y = rng.random(nobs) < special.expit(X @ np.array([0.5, 1.5, -1.0, 0.8, 0.0])); tau = 3.0
    else:
        from sklearn.datasets import load_breast_cancer
        D = load_breast_cancer(); cols = [0, 1, 4, 6]          # mean radius, texture, smoothness, concavity
        Xr = D.data[:, cols]; Xr = (Xr - Xr.mean(0)) / Xr.std(0)
        X = np.column_stack([np.ones(len(Xr)), Xr]); y = D.target == 1; p = X.shape[1]; tau = 2.5
    def logpost(b):
        if len(b) > 20000:
            return np.concatenate([logpost(b[i:i + 20000]) for i in range(0, len(b), 20000)])
        eta = b @ X.T; return (y * eta - np.logaddexp(0, eta)).sum(1) - 0.5 * np.sum(b ** 2, 1) / tau ** 2 - 0.5 * p * np.log(2 * np.pi * tau ** 2)
    mapb = optimize.minimize(lambda b: -logpost(b[None])[0], np.zeros(p), method="BFGS").x
    pr = special.expit(X @ mapb); Hm = (X.T * (pr * (1 - pr))) @ X + np.eye(p) / tau ** 2
    L = np.linalg.cholesky(np.linalg.inv(Hm)); ldL = np.log(np.diag(L)).sum(); lmax = logpost(mapb[None])[0]
    logp = lambda z: logpost(mapb + z @ L.T) + ldL - lmax          # rescaled (scale invariance)
    j = 1; fcoef = lambda z: (mapb + z @ L.T)[:, j]                  # posterior mean of beta_1
    def crit(nu, snis, M=200000, seed=1):
        z = nu + np.random.default_rng(seed).standard_normal((M, p)); lw = logp(z) - logq_iso(z, nu)
        w = np.exp(lw - lw.max())
        if not snis:
            return w.var() / w.mean() ** 2
        I = np.sum(w * fcoef(z)) / w.sum(); return np.mean(w ** 2 * (fcoef(z) - I) ** 2) / w.mean() ** 2
    # reference posterior sd of beta_1 for scaling the SNIS criterion
    zr = np.random.default_rng(0).standard_normal((400000, p)); lwr = logp(zr) - logq_iso(zr, np.zeros(p)); wr = np.exp(lwr - lwr.max())
    m1 = np.sum(wr * fcoef(zr)) / wr.sum(); v1 = np.sum(wr * (fcoef(zr) - m1) ** 2) / wr.sum()
    res = {"n_obs": int(len(y)), "p": p, "evidence_laplace": float(crit(np.zeros(p), False)),
           "snis_laplace": float(crit(np.zeros(p), True) / v1), "post_sd_b1": float(np.sqrt(v1))}
    n, B, reps = 20000, 100, 10
    for task in ["evidence", "snis"]:
        sn = task == "snis"
        for m in ["SN-AIS", "ce", "newton", "oais:0.3"]:
            vals = []
            for r in range(reps):
                tg = Target(logp, f=fcoef if sn else None, snis=sn)
                out = run_mean(m, tg, p, n, B, 50 + r)
                vals.append(crit(out["nus"][-1], sn, M=100000, seed=60 + r) / (v1 if sn else 1.0))
            res[f"{task}|{m}"] = {"median": float(np.median(vals)), "max": float(np.max(vals))}
            print(ARG, task, m, res[f"{task}|{m}"], flush=True)
    print(res); R_["E4"] = {ARG: res}

json.dump(R_, open(f"r2_{STAGE}{ARG}.json", "w"), default=float, indent=1)
print("done", STAGE, ARG)
