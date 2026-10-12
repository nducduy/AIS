"""SN-AIS (version 2) and baselines.  Whitened coordinates z; proposals N(nu, I) (mean
adaptation) or N(mu, diag(sig2)) in natural parameters (mean + diagonal covariance).
Objective  V = E_q[psi],  psi = F/q^2,  F = f^2 p^2  (IS)   or  F = (f - I)^2 p^2  (self-normalised IS).
"""
import numpy as np
LOG2PI = np.log(2 * np.pi)

def logq_iso(z, nu):
    return -0.5 * np.sum((z - nu) ** 2, 1) - 0.5 * z.shape[1] * LOG2PI

def logq_diag(z, mu, s2):
    return -0.5 * np.sum((z - mu) ** 2 / s2 + np.log(s2), 1) - 0.5 * z.shape[1] * LOG2PI

def _logmeanexp(a):
    """log of the mean of exp(a) over ALL entries (entries equal to -inf contribute zero)."""
    n = len(a); a = a[np.isfinite(a)]
    if len(a) == 0:
        return -np.inf
    m = a.max(); return m + np.log(np.sum(np.exp(a - m)) / n)

class Target:
    """logp(z): unnormalised log target in z-space; f(z): integrand values (or None for f = 1);
    snis: use the self-normalised objective with plug-in running estimate of I."""
    def __init__(self, logp, f=None, logf2=None, snis=False):
        self.logp, self.f, self.logf2, self.snis = logp, f, logf2, snis
        self.num = 0.0; self.den = 0.0; self.logscale = None      # running SNIS sums (rescaled)
    def lpsi(self, z, lw):
        if self.snis:
            fz = self.f(z)
            Ihat = self.num / self.den if self.den > 0 else np.sum(np.exp(lw - lw.max()) * fz) / np.sum(np.exp(lw - lw.max()))
            l2 = 2 * np.log(np.abs(fz - Ihat) + 1e-300)
        elif self.logf2 is not None:
            l2 = self.logf2(z)
        elif self.f is not None:
            l2 = 2 * np.log(np.abs(self.f(z)) + 1e-300)
        else:
            l2 = np.zeros(len(z))
        return l2 + 2 * lw
    def update_snis(self, z, lw):
        if not self.snis:
            return
        m = lw.max()
        if self.logscale is None:
            self.logscale = m
        if m > self.logscale:                       # rescale accumulated sums
            r = np.exp(self.logscale - m); self.num *= r; self.den *= r; self.logscale = m
        w = np.exp(lw - self.logscale); fz = self.f(z)
        self.num += np.sum(w * fz); self.den += np.sum(w)

def sm_update(S, v):
    Sv = S @ v; S -= np.outer(Sv, Sv) / (1.0 + v @ Sv)

# ---------------------------------------------------------------- SN-AIS v2 (mean adaptation)
SAFE_COUNT = [0]                      # number of times the safeguard was used (all runs)

def sn_mean(tg, d, n_total, B, rng, nu0=None, rho=2.0, gam=2.0, eig_lo=1e-8, eig_hi=1e8, clipM=50.0,
            record=None):
    """Algorithm 1.  c_t: weighted running average (weights k^rho) of past batch means of psi;
    the step uses the Hessian estimate A_t, which includes the current batch (self-damping).
    Safeguard: if the cheap bounds m_t <= lambda_min(A_t), tr(A_t) >= lambda_max(A_t) do not certify
    that the spectrum of A_t lies in [eig_lo, eig_hi], the eigenvalues of A_t are clipped to that interval."""
    nu = np.zeros(d) if nu0 is None else np.array(nu0, float)
    S = None; W = 0.0; k = 0; logC = None; logc1 = None; mbar = 0.0; trA = 0.0
    nus = [nu.copy()]; ests = []; n = 0; n_safe = 0
    while n < n_total:
        z = nu + rng.standard_normal((B, d)); lw = tg.logp(z) - logq_iso(z, nu)
        lpsi = tg.lpsi(z, lw); tg.update_snis(z, lw); ests.append((z, lw)); n += B
        lm = _logmeanexp(lpsi)
        if not np.isfinite(lm):
            nus.append(nu.copy()); continue
        k += 1
        if logC is None:
            logC = lm; logc1 = lm
        logc = min(max(logC, logc1 - clipM), logc1 + clipM)
        psi = np.exp(np.where(np.isfinite(lpsi), lpsi - logc, -np.inf)); s = z - nu
        g = (psi[:, None] * s).mean(0)                              # = -grad V / c
        wk = k ** rho; Wn = W + wk; beta = wk / Wn
        def hess_update(S):
            S = np.eye(d) * 1e8 if S is None else S / (1.0 - beta)
            fac = beta / B
            for i in np.nonzero(psi > 0)[0]:
                sm_update(S, np.sqrt(fac * psi[i]) * s[i])
            v = np.sqrt(fac * psi.sum())
            for j in range(d):
                e = np.zeros(d); e[j] = v; sm_update(S, e)
            return S
        mb_new = (1 - beta) * mbar + beta * psi.mean()               # lambda_min(A) >= mb
        tr_new = (1 - beta) * trA + beta * np.mean(psi * (np.sum(s ** 2, 1) + d))
        S = hess_update(S); mbar, trA = mb_new, tr_new        # A_t includes the current batch (self-damping)
        step = min(1.0, gam / (k + 1))
        if mbar >= eig_lo and trA + d * 1e-8 / Wn <= eig_hi:      # spectrum of A_t certified in [eig_lo, eig_hi]
            nu = nu + step * (S @ g)
        else:                                                         # safeguard: clip the spectrum of A_t
            ev, U = np.linalg.eigh(0.5 * (S + S.T))
            lamA = np.clip(1.0 / np.maximum(ev, 1e-300), eig_lo, eig_hi)
            nu = nu + step * (U @ ((U.T @ g) / lamA)); n_safe += 1; SAFE_COUNT[0] += 1
        W = Wn
        logC = np.logaddexp(np.log(1 - beta) + logC, np.log(beta) + lm) if k > 1 else lm
        nus.append(nu.copy())
    return {"nus": np.array(nus), "batches": ests, "n_safe": n_safe}

# ---------------------------------------------------------------- SN-AIS with diagonal covariance
def sn_diag(tg, d, n_total, B, rng, mu0=None, s20=None, rho=2.0, gam=2.0, max_ratio=4.0):
    """Natural parameters th1 = mu/s2, th2 = -1/(2 s2) per coordinate (p = 2d).
    Hessian of V: E[psi (s s^T + F)], F = Fisher (block-diagonal 2x2), inverse kept by Sherman-Morrison."""
    mu = np.zeros(d) if mu0 is None else np.array(mu0, float)
    s2 = np.ones(d) if s20 is None else np.array(s20, float)
    p = 2 * d; S = None; W = 0.0; k = 0; logC = None; n = 0
    hist = [(mu.copy(), s2.copy())]; ests = []
    while n < n_total:
        z = mu + np.sqrt(s2) * rng.standard_normal((B, d)); lw = tg.logp(z) - logq_diag(z, mu, s2)
        lpsi = tg.lpsi(z, lw); tg.update_snis(z, lw); ests.append((z, lw)); n += B
        lm = _logmeanexp(lpsi)
        if not np.isfinite(lm):
            hist.append((mu.copy(), s2.copy())); continue
        k += 1
        if logC is None:
            logC = lm
        psi = np.exp(np.where(np.isfinite(lpsi), lpsi - logC, -np.inf))
        # score in natural parameters: T - E T, T = (z, z^2)
        sc = np.concatenate([z - mu, z ** 2 - (mu ** 2 + s2)], 1)
        g = (psi[:, None] * sc).mean(0)
        wk = k ** rho; Wn = W + wk; beta = wk / Wn
        def hess_update(S):
            S = np.eye(p) * 1e8 if S is None else S / (1.0 - beta)
            fac = beta / B
            for i in np.nonzero(psi > 0)[0]:
                sm_update(S, np.sqrt(fac * psi[i]) * sc[i])
            tot = fac * psi.sum()
            for j in range(d):          # Fisher block for coordinate j: Cov(z, z^2)
                a = s2[j]; b = 2 * mu[j] * s2[j]; c = 2 * s2[j] ** 2 + 4 * mu[j] ** 2 * s2[j]
                l11 = np.sqrt(a); l21 = b / l11; l22 = np.sqrt(max(c - l21 ** 2, 1e-300))
                v1 = np.zeros(p); v1[j] = l11; v1[d + j] = l21
                v2 = np.zeros(p); v2[d + j] = l22
                sm_update(S, np.sqrt(tot) * v1); sm_update(S, np.sqrt(tot) * v2)
            return S
        S = hess_update(S)
        step = min(1.0, gam / (k + 1))
        th1 = mu / s2; th2 = -0.5 / s2
        delta = step * (S @ g)
        for _ in range(30):             # keep the variances positive and within a factor max_ratio
            n2 = th2 + delta[d:]
            if np.all(n2 < 0):
                s2n = -0.5 / n2
                if np.all(s2n <= max_ratio * s2) and np.all(s2n >= s2 / max_ratio):
                    break
            delta *= 0.5
        th1 = th1 + delta[:d]; th2 = th2 + delta[d:]
        s2 = -0.5 / th2; mu = th1 * s2
        W = Wn
        logC = np.logaddexp(np.log(1 - beta) + logC, np.log(beta) + lm) if k > 1 else lm
        hist.append((mu.copy(), s2.copy()))
    return {"hist": hist, "batches": ests}

# ---------------------------------------------------------------- baselines
def baseline(method, tg, d, n_total, B, rng, nu0=None, gamma0=1.0, R=10.0):
    """'ce'   : cross-entropy / moment matching of the mean (|f| p, or p if f is None);
       'newton': batch Newton (exact sample Hessian, O(d^3) per iteration);
       'nsgd' : normalised SGD (c_t of previous batch, step gamma0*min(1, 2/(t+1)));
       'oais' : Akyildiz-Miguez-type projected SGD on V, step gamma0/sqrt(t), normalised by the
                first-batch estimate, iterates projected on the ball of radius R, Polyak averaging."""
    nu = np.zeros(d) if nu0 is None else np.array(nu0, float); avg = nu.copy()
    nus = [nu.copy()]; ests = []; n = 0; t = 0; logc = None; logc0 = None
    while n < n_total:
        z = nu + rng.standard_normal((B, d)); lw = tg.logp(z) - logq_iso(z, nu)
        lpsi = tg.lpsi(z, lw); tg.update_snis(z, lw); ests.append((z, lw)); n += B; t += 1
        fin = np.isfinite(lpsi)
        if not fin.any():
            nus.append(avg.copy() if method == "oais" else nu.copy()); continue
        lm = _logmeanexp(lpsi)
        if method == "ce":
            a = 0.5 * lpsi[fin]; w = np.exp(a - a.max())
            nu = (w[:, None] * z[fin]).sum(0) / w.sum()
        elif method == "newton":
            psi = np.exp(np.where(fin, lpsi - lm, -np.inf)); s = z - nu
            H = (psi[:, None] * s).T @ s + psi.sum() * np.eye(d); nu = nu + np.linalg.solve(H, (psi[:, None] * s).sum(0))
        elif method == "nsgd":
            if logc is None:
                logc = lm
            psi = np.exp(np.where(fin, lpsi - logc, -np.inf))
            nu = nu + gamma0 * min(1.0, 2.0 / (t + 1)) * (psi[:, None] * (z - nu)).mean(0); logc = lm
        elif method == "oais":
            if logc0 is None:
                logc0 = lm
            psi = np.exp(np.where(fin, lpsi - logc0, -np.inf))
            nu = nu + gamma0 / np.sqrt(t) * (psi[:, None] * (z - nu)).mean(0)
            r = np.linalg.norm(nu)
            if not np.isfinite(r):
                nu = avg.copy()
            elif r > R:
                nu = nu * R / r
            avg = avg + (nu - avg) / t
        nus.append(avg.copy() if method == "oais" else nu.copy())
    return {"nus": np.array(nus), "batches": ests}

def ais_estimate(batches, f=None, logf2=None, snis=False):
    """Adaptive IS estimate from all stored batches (unnormalised: mean of f w; snis: ratio)."""
    lws = np.concatenate([b[1] for b in batches]); zs = np.concatenate([b[0] for b in batches])
    if snis:
        m = lws.max(); w = np.exp(lws - m); return np.sum(w * f(zs)) / np.sum(w)
    if logf2 is not None:
        return np.mean(np.exp(0.5 * logf2(zs) + lws))
    fz = f(zs) if f is not None else 1.0
    return np.mean(fz * np.exp(lws))
