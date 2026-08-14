"""Descriptive facts for Bitcoin OTC (research plan section 8.3).

Covers: event/entry rates, out/in heterogeneity, sign dynamics over time,
capital distribution shape (skew, tail, Gini), target-selection dependence
(the G1 identifiability probe), cross-group mixing, and the impact of the
once-per-pair constraint on risk-set size.

Capital model (research plan 7.1/7.3): each received rating is a jump of
magnitude ``eta*r`` (we use ``r/10`` so single ratings live in [-1,1]) and
capital decays multiplicatively with rate ``delta = ln(2)/half_life`` between
events.  Capital types:
  - count : decayed number of received ratings (jump +1)
  - net   : decayed sum of normalized ratings (jump r/10)
  - quality: smoothed ratio net/(count+1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from .data import sort_by_time

LN2 = np.log(2.0)
ETA = 1.0 / 10.0  # normalize ratings to [-1, 1]


# --------------------------------------------------------------------------
# attention engine (H state: exponentially-decayed incoming-event count)
# --------------------------------------------------------------------------

class AttentionEngine:
    """Lazily-decayed per-node attention memory with unit incoming jumps.

    H_j(t) = sum over incoming events m< t of exp(-rho * (t - tau_m)),
    i.e. dH_j = -rho H_j dt + dN_j^in.  Same lazy-decay pattern as
    CapitalEngine, but every incoming event adds 1.0 rather than a mark.
    """

    def __init__(self, n_nodes: int, rho: float):
        self.rho = rho
        self.value = np.zeros(n_nodes)
        self.last_t = np.full(n_nodes, -np.inf)

    def at(self, node: int, t: float) -> float:
        if t > self.last_t[node]:
            v = self.value[node]
            if v != 0.0:  # guard: 0 * exp(-rho * inf) would be NaN
                self.value[node] = v * np.exp(-self.rho * (t - self.last_t[node]))
            self.last_t[node] = t
        return self.value[node]

    def at_all(self, t: float) -> np.ndarray:
        mask = (self.value != 0.0) & (t > self.last_t)
        if np.any(mask):
            self.value[mask] *= np.exp(-self.rho * (t - self.last_t[mask]))
            self.last_t[mask] = t
        return self.value

    def jump(self, node: int, t: float) -> None:
        """Incoming event at time t: decay to t first, then add 1.0."""
        self.at(node, t)
        self.value[node] += 1.0


# --------------------------------------------------------------------------
# capital engine
# --------------------------------------------------------------------------

class CapitalEngine:
    """Lazily-decayed per-node capital with additive marked jumps.

    State per node: value, last time it was decayed.  Access at time t
    decays on the fly; updates decay first, then add the jump.
    """

    def __init__(self, n_nodes: int, delta: float):
        self.delta = delta
        self.value = np.zeros(n_nodes)
        self.last_t = np.full(n_nodes, -np.inf)

    def _decay(self, node: int, t: float) -> None:
        v = self.value[node]
        if v != 0.0 and t > self.last_t[node]:
            self.value[node] = v * np.exp(-self.delta * (t - self.last_t[node]))
        self.last_t[node] = t

    def at(self, node: int, t: float) -> float:
        if t > self.last_t[node]:
            v = self.value[node]
            if v != 0.0:   # guard: 0 * exp(-0 * inf) would be NaN (delta = 0)
                self.value[node] = v * np.exp(-self.delta * (t - self.last_t[node]))
            self.last_t[node] = t
        return self.value[node]

    def at_all(self, t: float) -> np.ndarray:
        """Decay every node to ``t`` and return all values (vectorized)."""
        mask = (self.value != 0.0) & (t > self.last_t)
        if np.any(mask):
            self.value[mask] *= np.exp(-self.delta * (t - self.last_t[mask]))
            self.last_t[mask] = t
        return self.value

    def jump(self, node: int, t: float, amount: float) -> None:
        self._decay(node, t)
        self.value[node] += amount


def compute_capitals(df: pd.DataFrame, half_life_days: float | None,
                     cap_type: str = "net") -> pd.DataFrame:
    """Per-event capital snapshots: returns a copy of df with extra columns
    ``cap_source`` / ``cap_target`` (capital of each side just before the
    event) and ``cap_after`` (target capital just after the event)."""
    df = sort_by_time(df).copy()
    nodes = np.unique(np.concatenate([df["source"].values, df["target"].values]))
    idx = {n: k for k, n in enumerate(nodes)}
    delta = LN2 / (half_life_days * 86400.0) if half_life_days else 0.0
    n_events = len(df)
    cap_src = np.zeros(n_events)
    cap_tgt = np.zeros(n_events)
    cap_after = np.zeros(n_events)
    if cap_type == "quality":
        net_eng, cnt_eng = CapitalEngine(len(nodes), delta), CapitalEngine(len(nodes), delta)
        for e, row in enumerate(df.itertuples(index=False)):
            i, j, r, t = row.source, row.target, row.rating, row.timestamp
            cap_src[e] = net_eng.at(idx[i], t) / (cnt_eng.at(idx[i], t) + 1.0)
            cap_tgt[e] = net_eng.at(idx[j], t) / (cnt_eng.at(idx[j], t) + 1.0)
            net_eng.jump(idx[j], t, ETA * r)
            cnt_eng.jump(idx[j], t, 1.0)
            cap_after[e] = net_eng.at(idx[j], t) / (cnt_eng.at(idx[j], t) + 1.0)
    else:
        eng = CapitalEngine(len(nodes), delta)
        jump = 1.0 if cap_type == "count" else ETA
        for e, row in enumerate(df.itertuples(index=False)):
            i, j, r, t = row.source, row.target, row.rating, row.timestamp
            cap_src[e] = eng.at(idx[i], t)
            cap_tgt[e] = eng.at(idx[j], t)
            eng.jump(idx[j], t, jump * r if cap_type == "net" else jump)
            cap_after[e] = eng.value[idx[j]]
    df["cap_source"] = cap_src
    df["cap_target"] = cap_tgt
    df["cap_after"] = cap_after
    return df


def capital_distribution_snapshots(
    df: pd.DataFrame, nodes: np.ndarray, entry: pd.Series,
    delta: float, cap_type: str = "net", n_bins: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Capital (decayed) of all *active* nodes at ``n_bins`` evenly spaced
    sample times.  Returns (times, capital matrix of shape (n_bins, n_nodes));
    inactive nodes at a snapshot are NaN."""
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    times = np.linspace(t0, t1, n_bins + 1)[:-1]
    eng = CapitalEngine(len(nodes), delta)
    cnt_eng = CapitalEngine(len(nodes), delta) if cap_type == "quality" else None
    j_idx = entry["entry"].reindex(nodes).values  # entry times, aligned to nodes
    pos = {n: k for k, n in enumerate(nodes)}
    out = np.full((len(times), len(nodes)), np.nan)
    events = sort_by_time(df).itertuples(index=False)
    ev = next(events, None)
    for k, t in enumerate(times):
        while ev is not None and ev.timestamp <= t:
            pj = pos[ev.target]
            if cap_type == "quality":
                eng.jump(pj, ev.timestamp, ETA * ev.rating)
                cnt_eng.jump(pj, ev.timestamp, 1.0)
            else:
                eng.jump(pj, ev.timestamp,
                         1.0 if cap_type == "count" else ETA * ev.rating)
            ev = next(events, None)
        active = j_idx <= t
        vals = eng.at_all(t)
        if cap_type == "quality":
            vals = vals / (cnt_eng.at_all(t) + 1.0)
        out[k, active] = vals[active]
    return times, out


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if len(x) == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) * x).sum() / (n * x.sum()) - (n + 1) / n)


def distribution_shape(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {}
    s = np.sort(x)
    n = len(x)
    top1 = s[int(0.99 * n):].sum() / s.sum() if s.sum() > 0 else np.nan
    top10 = s[int(0.90 * n):].sum() / s.sum() if s.sum() > 0 else np.nan
    return {
        "n": int(n),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "skewness": float(skew(x)),
        "kurtosis": float(kurtosis(x)),
        "gini": gini(x),
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "top1pct_share": float(top1),
        "top10pct_share": float(top10),
        "n_negative": int((x < 0).sum()),
        "n_zero": int((x == 0).sum()),
    }


# --------------------------------------------------------------------------
# target-selection probe (case-control conditional logistic; the G1 probe)
# --------------------------------------------------------------------------

def build_target_case_control(
    df: pd.DataFrame, entry: pd.Series, n_controls: int = 30,
    rng: np.random.Generator | None = None, max_events: int | None = None,
) -> pd.DataFrame:
    """For each event (i -> j, t): covariates of the chosen target j and of
    ``n_controls`` random *active* non-chosen candidates at time t.

    Returns one row per candidate (case = 1 for the chosen target) with
    time-varying covariates:
      log_indegree : log(1 + number of ratings received so far)
      net_cap      : decayed net capital of the candidate at t (half-life 180d)
      count_cap    : decayed count capital (same half-life)
      age_days     : time since entry, in days
      log_recency  : log(1 + seconds since candidate last received a rating)
      ever_sent    : whether the candidate has ever sent a rating (activity proxy)
      sent_count   : number of ratings sent so far
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    df = sort_by_time(df).copy()
    if max_events is not None:
        df = df.head(max_events)
    nodes = np.unique(np.concatenate([df["source"].values, df["target"].values]))
    n = len(nodes)
    pos = {nd: k for k, nd in enumerate(nodes)}
    entry_t = entry["entry"].reindex(nodes).values

    half_life = 180.0
    delta = LN2 / (half_life * 86400.0)
    net = CapitalEngine(n, delta)
    cnt = CapitalEngine(n, delta)
    indeg = np.zeros(n, dtype=int)
    sent = np.zeros(n, dtype=int)
    last_recv = np.full(n, -np.inf)
    active_since = np.zeros(n, dtype=bool)

    rows = []
    for e, row in enumerate(df.itertuples(index=False)):
        i, j, r, t = row.source, row.target, row.rating, row.timestamp
        pi, pj = pos[i], pos[j]
        active_since[pi] = True
        active_since[pj] = True
        # candidates: chosen target + sampled active others
        active = np.flatnonzero(active_since)
        others = active[active != pj]
        if len(others) < n_controls:
            ctrl = others
        else:
            ctrl = rng.choice(others, size=n_controls, replace=False)
        cand = np.concatenate([[pj], ctrl])
        case = np.zeros(len(cand))
        case[0] = 1.0

        net_at = np.array([net.at(c, t) for c in cand])
        cnt_at = np.array([cnt.at(c, t) for c in cand])
        age = (t - entry_t[cand]) / 86400.0
        rec_raw = (t - last_recv[cand]) / 86400.0
        rec = np.where(np.isfinite(rec_raw), rec_raw, age)  # never received -> node age
        ever_sent = (sent[cand] > 0).astype(float)
        rows.append(pd.DataFrame({
            "event": e, "stratum": e, "case": case,
            "log_indegree": np.log1p(indeg[cand]),
            "net_cap": net_at, "count_cap": cnt_at,
            "age_days": age, "log_recency": np.log1p(np.maximum(rec, 0.0)),
            "ever_sent": ever_sent, "sent_count": sent[cand],
        }))
        # update state with the realized event
        indeg[pj] += 1
        sent[pi] += 1
        cnt.jump(pj, t, 1.0)
        net.jump(pj, t, ETA * r)
        last_recv[pj] = t
    return pd.concat(rows, ignore_index=True)


def run_g1_probe(df: pd.DataFrame, entry: pd.Series,
                 max_events: int = 12000, n_controls: int = 30,
                 seed: int = 0) -> dict:
    """Conditional logistic regression of 'being chosen' on candidate
    covariates; returns coefficients, standard errors and a negative-control
    (shuffled-target) comparison."""
    from statsmodels.discrete.conditional_models import ConditionalLogit

    cc = build_target_case_control(df, entry, n_controls=n_controls,
                                   max_events=max_events,
                                   rng=np.random.default_rng(seed))
    feats = ["log_indegree", "net_cap", "count_cap", "age_days",
             "log_recency", "ever_sent"]
    X = cc[feats]
    y = cc["case"]
    model = ConditionalLogit(endog=y, exog=X, groups=cc["stratum"])
    res = model.fit(disp=False, maxiter=200)
    coef = {f: float(res.params[f]) for f in feats}
    se = {f: float(res.bse[f]) for f in feats}
    pval = {f: float(res.pvalues[f]) for f in feats}

    # negative control: shuffle the case flag within each stratum
    rng = np.random.default_rng(seed + 1)
    cc2 = cc.copy()
    cc2["case"] = cc2.groupby("stratum")["case"].transform(
        lambda s: rng.permutation(s.values))
    model2 = ConditionalLogit(endog=cc2["case"], exog=X, groups=cc2["stratum"])
    res2 = model2.fit(disp=False, maxiter=200)
    coef_neg = {f: float(res2.params[f]) for f in feats}

    return {
        "n_events": int(cc["event"].nunique()),
        "n_rows": int(len(cc)),
        "coef": coef, "se": se, "pvalue": pval,
        "coef_negative_control": coef_neg,
    }


# --------------------------------------------------------------------------
# rates, heterogeneity, mixing
# --------------------------------------------------------------------------

def monthly_rates(df: pd.DataFrame) -> pd.DataFrame:
    from .data import node_entry_times

    df = sort_by_time(df)
    ts = pd.to_datetime(df["timestamp"], unit="s")
    ev = ts.groupby(ts.dt.to_period("M")).size().rename("events")
    entry = node_entry_times(df)
    ent = (pd.to_datetime(entry["entry"], unit="s")
           .dt.to_period("M").value_counts().sort_index().rename("new_users"))
    out = pd.concat([ev, ent], axis=1).fillna(0)
    out["active_users"] = out["new_users"].cumsum()
    return out


def heterogeneity(df: pd.DataFrame) -> dict:
    out = df.groupby("source").size()
    inn = df.groupby("target").size()
    both = pd.DataFrame({"out": out, "in": inn}).fillna(0)
    both["total"] = both["out"] + both["in"]
    out_g = gini(both["out"].values)
    in_g = gini(both["in"].values)
    corr = float(np.corrcoef(both["out"].values, both["in"].values)[0, 1])
    return {
        "n_users": int(len(both)),
        "out_mean": float(both["out"].mean()),
        "out_p90": float(both["out"].quantile(0.9)),
        "out_p99": float(both["out"].quantile(0.99)),
        "out_max": float(both["out"].max()),
        "out_gini": out_g,
        "in_mean": float(both["in"].mean()),
        "in_p90": float(both["in"].quantile(0.9)),
        "in_p99": float(both["in"].quantile(0.99)),
        "in_max": float(both["in"].max()),
        "in_gini": in_g,
        "corr_out_in": corr,
        "share_users_never_sent": float((both["out"] == 0).mean()),
        "share_users_never_received": float((both["in"] == 0).mean()),
    }


def sign_series(df: pd.DataFrame) -> pd.DataFrame:
    df = sort_by_time(df).copy()
    df["month"] = pd.to_datetime(df["timestamp"], unit="s").dt.to_period("M")
    g = df.groupby("month")["rating"].agg(["size", "mean"])
    neg = df[df["rating"] < 0].groupby("month").size().rename("n_neg")
    out = g.join(neg).fillna(0)
    out["neg_share"] = out["n_neg"] / out["size"]
    out["mean_rating"] = out["mean"]
    return out.drop(columns=["mean"])


def mixing_matrix(df: pd.DataFrame, k: int = 5, seed: int = 0) -> dict:
    """Spectral clustering of the symmetrized training adjacency into ``k``
    latent groups, then the directed mixing table and its lift vs random."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    df = sort_by_time(df)
    nodes = np.unique(np.concatenate([df["source"].values, df["target"].values]))
    pos = {nd: kk for kk, nd in enumerate(nodes)}
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
    group = np.full(len(nodes), -1)
    group[keep] = lab
    M = np.zeros((k, k))
    for row in df.itertuples(index=False):
        gs, gt = group[pos[row.source]], group[pos[row.target]]
        if gs >= 0 and gt >= 0:
            M[gs, gt] += 1.0
    out = M.sum(axis=1)
    inn = M.sum(axis=0)
    tot = M.sum()
    E = np.outer(out, inn) / tot if tot else np.zeros_like(M)
    lift = np.divide(M, E, out=np.zeros_like(M), where=E > 0)
    return {
        "k": k,
        "group_sizes": [int((group == g).sum()) for g in range(k)],
        "mixing": M.tolist(),
        "expected": E.tolist(),
        "lift": lift.tolist(),
        "homophily_lift": float(np.diag(lift).mean()) if k else np.nan,
        "share_diag": float(np.diag(M).sum() / tot) if tot else np.nan,
    }
