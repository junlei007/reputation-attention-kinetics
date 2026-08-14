"""Staged estimation pipeline for Bitcoin OTC (research plan section 8.4).

Stages (each documented in notes/WP5_estimation.md):
  1. activity        : per-node out-event rate over active time, shrunk
  2. latent types     : spectral clustering of the symmetrised training graph
  3. kernel W         : block MLE with exposure, two-stage (psi refinement)
  4. capital response : conditional-logit profile over the tanh parameterisation
                       psi2(c) = 1 + alpha*tanh(beta*c)  (bounded, sign-free)
  5. marks Q          : sign logistic + empirical magnitude (by capital bin)
  6. decay delta      : profile over half-lives via the MACRO-level prediction
                       (event-level likelihoods are nearly delta-flat, WP2 G2)
  7. rate calibration : the absolute scale is identified by the training
                       event count (pair rates are (1/N)-scaled with W and psi
                       normalised); a single constant is calibrated to the
                       training total rate.

Scale conventions: W_hat is normalised to mean 1 over block pairs; psi2(0)=1;
the calibrated constant c_rate absorbs the remaining scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import node_entry_times, sort_by_time
from .facts import AttentionEngine, CapitalEngine, LN2, ETA


# --------------------------------------------------------------------------
# 1. activity
# --------------------------------------------------------------------------

def fit_activity(df: pd.DataFrame, entry: pd.Series | None = None,
                 shrink: float = 2.0) -> pd.DataFrame:
    """Per-node out-event rate a_hat = (out + shrink) / (T_active + shrink)
    over the observed span; returns a Series indexed by node."""
    df = sort_by_time(df)
    if entry is None:
        entry = node_entry_times(df)
    _, t1 = df["timestamp"].min(), df["timestamp"].max()
    out = df.groupby("source").size()
    active_t = np.maximum((t1 - entry["entry"]) / 86400.0, 0.0)
    out = out.reindex(entry.index, fill_value=0)
    rate = (out + shrink) / (active_t + shrink)
    return rate.rename("a_hat")


# --------------------------------------------------------------------------
# 2. latent types (spectral clustering)
# --------------------------------------------------------------------------

def fit_types(df: pd.DataFrame, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Spectral clustering of the symmetrised directed adjacency; returns a
    Series of group labels indexed by node (only nodes with degree > 0)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    df = sort_by_time(df)
    nodes = np.unique(np.concatenate([df["source"].values, df["target"].values]))
    pos = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for row in df.itertuples(index=False):
        A[pos[row.source], pos[row.target]] += 1.0
    As = A + A.T
    deg = As.sum(axis=1)
    keep = deg > 0
    L = normalize(As[keep][:, keep], norm="l2", axis=1)
    U, S, _ = np.linalg.svd(L, full_matrices=False)
    embed = U[:, :k] * S[:k]
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embed)
    groups = pd.Series(-1, index=nodes, dtype=int)
    groups.iloc[np.flatnonzero(keep)] = lab
    return groups.rename("group")


# --------------------------------------------------------------------------
# 3. block kernel
# --------------------------------------------------------------------------

def fit_kernel(df: pd.DataFrame, groups: pd.Series, a_hat: pd.Series,
               psi1_vals: np.ndarray | None = None,
               psi2_vals: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Block MLE: W_hat[g,h] = count_gh / exposure_gh, where exposure_gh =
    (1/N) * T * (sum_{i in g} a_i psi1_i) * (sum_{j in h} psi2_j) over the
    active population; psi1/psi2 value vectors (aligned to nodes) default to 1.

    Returns (W_hat normalised to mean 1 over pairs with exposure > 0, N)."""
    df = sort_by_time(df)
    nodes = np.asarray(groups.index)
    pos = {n: i for i, n in enumerate(nodes)}
    g = groups.values.astype(int)
    K = int(groups.max()) + 1 if len(groups) else 0
    N = len(nodes)
    a_node = a_hat.reindex(nodes).fillna(0).to_numpy()
    if psi1_vals is None:
        psi1_vals = np.ones(N)
    else:
        psi1_vals = np.asarray(psi1_vals, dtype=float)
    if psi2_vals is None:
        psi2_vals = np.ones(N)
    else:
        psi2_vals = np.asarray(psi2_vals, dtype=float)
    # Sender exposure is the sum of the products, not the product of the
    # two group sums.  Multiplying sum(a_i) by sum(psi1_i) would introduce
    # an erroneous extra group-size factor when psi1 == 1.
    sender_g = np.bincount(g, weights=a_node * psi1_vals, minlength=K)
    target_g = np.bincount(g, weights=psi2_vals, minlength=K)
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    T = (t1 - t0) / 86400.0
    exposure = (1.0 / N) * T * sender_g[:, None] * target_g[None, :]
    count = np.zeros((K, K))
    for row in df.itertuples(index=False):
        gi = g[pos[row.source]]
        gj = g[pos[row.target]]
        if gi >= 0 and gj >= 0:
            count[gi, gj] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        W_hat = np.divide(count, exposure, out=np.zeros_like(count),
                          where=exposure > 0)
    m = exposure > 0
    scale = W_hat[m].mean() if m.any() else 1.0
    W_hat = np.divide(W_hat, scale, out=np.zeros_like(W_hat), where=scale > 0)
    return W_hat, N


def calibrate_rate(df: pd.DataFrame, groups: pd.Series, a_hat: pd.Series,
                   W_hat: np.ndarray,
                   psi1_vals: np.ndarray | None = None,
                   psi2_vals: np.ndarray | None = None) -> float:
    """Calibrate the global rate multiplier from the training event count.

    This uses the same block exposure as :func:`fit_kernel`, including the
    empirical group sizes and activities.  An unweighted ``W_hat.mean()``
    is not a valid substitute when block sizes are highly unequal.
    """
    nodes = np.asarray(groups.index)
    g = groups.to_numpy(dtype=int)
    n = len(nodes)
    k = int(groups.max()) + 1 if len(groups) else 0
    a_node = a_hat.reindex(nodes).fillna(0).to_numpy()
    p1 = np.ones(n) if psi1_vals is None else np.asarray(psi1_vals, dtype=float)
    p2 = np.ones(n) if psi2_vals is None else np.asarray(psi2_vals, dtype=float)
    sender_g = np.bincount(g, weights=a_node * p1, minlength=k)
    target_g = np.bincount(g, weights=p2, minlength=k)
    t_days = (df["timestamp"].max() - df["timestamp"].min()) / 86400.0
    exposure_count = (t_days / n) * float(sender_g @ W_hat @ target_g)
    return float(len(df) / max(exposure_count, 1e-300))


# --------------------------------------------------------------------------
# 4. case-control machinery (target-choice model)
# --------------------------------------------------------------------------

def build_case_control(df: pd.DataFrame, groups: pd.Series,
                       entry: pd.Series, delta: float,
                       a_hat: pd.Series | None = None,
                       W_hat: np.ndarray | None = None,
                       n_controls: int = 30, max_events: int | None = None,
                       seed: int = 0, cap_type: str = "net",
                       h_rho: float | None = None,
                       exclude_used: bool = False,
                       psi_h: tuple | None = None,
                       psi_exp_beta: float | None = None,
                       psi_c_beta: float | None = None) -> pd.DataFrame:
    """Per-event case-control dataset: the chosen target plus n_controls
    active non-chosen candidates, with time-varying covariates.  Reuses the
    G1-probe construction (facts.build_case_control-like) but on the real
    data with groups and the estimated kernel as controls."""
    rng = np.random.default_rng(seed)
    df = sort_by_time(df).copy()
    if max_events is not None:
        df = df.head(max_events)
    nodes = np.asarray(groups.index)
    n = len(nodes)
    pos = {nd: k for k, nd in enumerate(nodes)}
    g = groups.values
    entry_t = entry["entry"].reindex(nodes).values
    net = CapitalEngine(n, delta)
    cnt = CapitalEngine(n, delta)
    att = AttentionEngine(n, h_rho) if h_rho is not None else None
    used: dict[int, set[int]] = {}
    indeg = np.zeros(n, dtype=int)
    sent = np.zeros(n, dtype=int)
    last_recv = np.full(n, -np.inf)
    active_since = np.zeros(n, dtype=bool)
    a_arr = a_hat.reindex(nodes).fillna(0.0).values if a_hat is not None else None
    rows = []
    for e, row in enumerate(df.itertuples(index=False)):
        i, j, r, t = row.source, row.target, row.rating, row.timestamp
        pi_, pj = pos[i], pos[j]
        active_since[pi_] = True
        active_since[pj] = True
        act = np.flatnonzero(active_since)
        # legal risk set: the realised target plus active candidates that
        # are neither the sender nor an already-rated ordered pair
        others = act[(act != pj) & (act != pi_)]
        if exclude_used:
            others = np.array([c for c in others
                               if c not in used.get(pi_, set())])
        ctrl = rng.choice(others, size=min(n_controls, len(others)),
                          replace=False)
        cand = np.concatenate([[pj], ctrl]).astype(np.int64)
        case = np.zeros(len(cand)); case[0] = 1.0
        net_at = np.array([net.at(c, t) for c in cand])
        cnt_at = np.array([cnt.at(c, t) for c in cand])
        h_at = (np.array([att.at(c, t) for c in cand])
                if att is not None else None)
        age = (t - entry_t[cand]) / 86400.0
        rec_raw = (t - last_recv[cand]) / 86400.0
        rec = np.where(np.isfinite(rec_raw), rec_raw, age)
        ever_sent = (sent[cand] > 0).astype(float)
        logW = np.zeros(len(cand))
        if W_hat is not None:
            gi = g[pi_]
            logW = np.array([np.log(W_hat[gi, g[c]] + 1e-9)
                             if gi >= 0 and g[c] >= 0 else 0.0 for c in cand])
        feats = {
            "stratum": e, "case": case,
            "log_indegree": np.log1p(indeg[cand]),
            "net_cap": net_at, "count_cap": cnt_at,
            "age_days": age, "log_recency": np.log1p(np.maximum(rec, 0.0)),
            "ever_sent": ever_sent, "logW": logW,
        }
        if h_at is not None:
            feats["logH"] = np.log1p(h_at)
        if psi_h is not None:
            alpha_h, beta_h = psi_h
            # bounded, Lipschitz attention response: log(1 + a tanh(b H))
            feats["logPsiH"] = np.log1p(alpha_h * np.tanh(beta_h * h_at))
        if psi_exp_beta is not None:
            # bounded, Lipschitz response psi_H(H) = exp(g tanh(b H)):
            # the feature is tanh(b H) and the exponent g is estimated
            # freely by the conditional logit
            feats["tanhH"] = np.tanh(psi_exp_beta * h_at)
        if psi_c_beta is not None:
            # bounded, Lipschitz capital response psi_C(C) =
            # exp(g tanh(b C)) on the NET capital: feature tanh(b C),
            # exponent g free (theory-aligned capital channel)
            feats["tanhC"] = np.tanh(psi_c_beta * net_at)
        rows.append(pd.DataFrame(feats))
        indeg[pj] += 1
        sent[pi_] += 1
        cnt.jump(pj, t, 1.0)
        net.jump(pj, t, ETA * r)
        if att is not None:
            att.jump(pj, t)
        if exclude_used:
            used.setdefault(pi_, set()).add(pj)
        last_recv[pj] = t
    return pd.concat(rows, ignore_index=True)


def conditional_logit_fit(cc: pd.DataFrame, feats):
    from statsmodels.discrete.conditional_models import ConditionalLogit

    m = ConditionalLogit(endog=cc["case"], exog=cc[list(feats)],
                         groups=cc["stratum"])
    res = m.fit(disp=False, maxiter=300)
    return res


def fit_psi_profile(df: pd.DataFrame, groups: pd.Series, entry: pd.Series,
                    delta: float, a_hat: pd.Series, W_hat: np.ndarray,
                    alphas=(0.3, 0.6, 0.9, 0.99),
                    betas=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
                    n_controls: int = 20, max_events: int = 9000,
                    seed: int = 0) -> dict:
    """Profile ``psi2(c) = 1 + alpha*tanh(beta*c)`` on a nested holdout.

    The sender term is constant within each target-choice stratum.  For each
    grid point we therefore evaluate the structural target score
    ``log W + log psi2(c)`` on the final 30% of the (training-period)
    case-control strata and select its conditional log likelihood.  This
    direct profile is essential: putting only ``tanh(beta*c)`` into a
    conditional logit with a free coefficient makes ``alpha`` disappear and
    does not estimate the stated bounded response.
    """
    cc = build_case_control(df, groups, entry, delta,
                            a_hat=a_hat, W_hat=W_hat,
                            n_controls=n_controls,
                            max_events=max_events, seed=seed)
    n_strata = int(cc["stratum"].max()) + 1
    split = int(0.7 * n_strata)
    te = cc[cc["stratum"] >= split]
    cap = te["net_cap"].to_numpy()
    log_w = te["logW"].to_numpy()
    cases = te["case"].to_numpy(dtype=bool)
    strata = te["stratum"].to_numpy()
    results = {}
    for alpha in alphas:
        for beta in betas:
            score = log_w + np.log(np.maximum(
                1e-12, 1.0 + alpha * np.tanh(beta * cap)))
            ll = 0.0
            for s in np.unique(strata):
                take = strata == s
                z = score[take].copy()
                z = z - z.max()
                ll += z[cases[take]].sum() - np.log(np.exp(z).sum())
            results[(alpha, beta)] = ll
    best = max(results, key=results.get)
    return {"best_alpha": best[0], "best_beta": best[1],
            "profile": {f"a={a},b={b}": round(ll, 2)
                        for (a, b), ll in results.items()},
            "held_out_loglik_best": round(results[best], 2)}


def psi2_from(alpha: float, beta: float):
    def psi2(c):
        return 1.0 + alpha * np.tanh(beta * np.asarray(c, dtype=float))

    return psi2


# --------------------------------------------------------------------------
# 5. marks
# --------------------------------------------------------------------------

def fit_mark_models(df: pd.DataFrame, delta: float,
                    groups: pd.Series | None = None,
                    half_life_bins: int = 6) -> dict:
    """Sign model: logistic on (net_cap, count_cap, age_days, year-index,
    target group dummies); magnitude model: empirical |r| distribution by
    net-capital bin (truncated to 1..10)."""
    from sklearn.linear_model import LogisticRegression

    df = sort_by_time(df).copy()
    nodes = np.asarray(groups.index) if groups is not None else None
    pos = {n: k for k, n in enumerate(nodes)} if nodes is not None else {}
    net = CapitalEngine(len(nodes), delta) if nodes is not None else None
    cnt = CapitalEngine(len(nodes), delta) if nodes is not None else None
    t0 = df["timestamp"].min()
    rows = []
    for row in df.itertuples(index=False):
        i, j, r, t = row.source, row.target, row.rating, row.timestamp
        cj = net.at(pos[j], t) if net is not None else 0.0
        kj = cnt.at(pos[j], t) if cnt is not None else 0.0
        rows.append({
            "net_cap": cj, "count_cap": kj,
            "year_idx": (t - t0) / (86400.0 * 365.0),
            "rating": r,
        })
        if net is not None:
            net.jump(pos[j], t, ETA * r)
            cnt.jump(pos[j], t, 1.0)
    dat = pd.DataFrame(rows)
    X = dat[["net_cap", "count_cap", "year_idx"]].values
    y = (dat["rating"] > 0).astype(int)
    lr = LogisticRegression(C=1e6, max_iter=3000).fit(X, y)
    mag = dat["rating"].abs().to_numpy(dtype=int)
    bins = np.unique(np.quantile(
        dat["net_cap"], np.linspace(0, 1, half_life_bins + 1)))
    if len(bins) < 2:
        bins = np.array([float(dat["net_cap"].min()),
                         float(dat["net_cap"].max()) + 1e-9])
    bins[-1] += 1e-9
    global_mag = np.bincount(mag, minlength=11)[1:11].astype(float)
    global_mag /= global_mag.sum()
    bin_id = np.searchsorted(bins[1:-1], dat["net_cap"], side="right")
    mag_dist = []
    for b in range(len(bins) - 1):
        counts = np.bincount(mag[bin_id == b], minlength=11)[1:11].astype(float)
        mag_dist.append((counts / counts.sum() if counts.sum()
                         else global_mag).tolist())
    return {"sign_coef": lr.coef_[0].tolist(), "sign_intercept": float(lr.intercept_[0]),
            "mag_bin_edges": bins.tolist(), "mag_dist": mag_dist}


# --------------------------------------------------------------------------
# 6. decay selection via macro prediction
# --------------------------------------------------------------------------

def capitals_series(df: pd.DataFrame, nodes: np.ndarray, delta: float,
                    cap_type: str = "net") -> np.ndarray:
    """Capital of the target at each event time (realised-history filter)."""
    pos = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    C = np.zeros(n)
    last_t = np.full(n, -np.inf)
    out = np.zeros(len(df))
    for e, row in enumerate(df.itertuples(index=False)):
        j = pos[row.target]
        if row.timestamp > last_t[j]:
            C[j] *= np.exp(-delta * (row.timestamp - last_t[j]))
            last_t[j] = row.timestamp
        out[e] = C[j]
        C[j] += ETA * row.rating
        last_t[j] = row.timestamp
    return out


def select_delta(df: pd.DataFrame, grid_days=(30, 90, 180, 365, None),
                 seed: int = 0) -> dict:
    """Profile half-lives by the realised-history capital autocorrelation:
    for each delta, recompute the capital series and fit the sign model;
    the profile uses the training/test log-likelihood difference.  (Event
    choice likelihoods are nearly flat in delta, WP2 G2; the plan's delta
    treatment is profile + sensitivity.)"""
    from sklearn.linear_model import LogisticRegression

    df = sort_by_time(df)
    nodes = np.unique(np.concatenate([df["source"].values, df["target"].values]))
    n = len(df)
    split = int(0.7 * n)
    y = (df["rating"].values > 0).astype(int)
    t0 = df["timestamp"].min()
    results = {}
    for hl in grid_days:
        d = LN2 / (hl * 86400.0) if hl else 0.0
        C = capitals_series(df, nodes, d)
        X = np.column_stack([np.zeros(n), C,
                             (df["timestamp"].values - t0) / (86400 * 365)])
        lr = LogisticRegression(C=1e6, max_iter=3000)
        lr.fit(X[:split], y[:split])
        pte = lr.predict_proba(X[split:])
        ll = float(np.sum(np.log(pte[np.arange(len(pte)), y[split:]])))
        results[hl] = ll
    best = max(results, key=results.get)
    return {"best_hl_days": best, "profile": {str(k): round(v, 2)
                                              for k, v in results.items()}}
