#!/usr/bin/env python
"""Ablation: attention memory (H) vs capital (C) in target choice (WP6+).

Nested models on the case-control conditional-logit scale, all estimated on
the training period only and evaluated strictly out-of-sample on the test
period:

    M0: W                    (kernel only)
    M1: W + C                (kernel + net/count capital)
    M2: W + H                (kernel + attention memory, log1p(H))
    M3: W + C + H            (kernel + capital + attention)
    M4: g(t) * (W + C + H)   (macro: platform-level time effect)

Prespecified protocol (specification and criteria fixed before the
Alpha analysis; user review, 2026-08-12):
  * legal risk set used identically in training and test: active
    candidates excluding the sender and already-rated ordered pairs
    (previously the training case-control did not exclude used pairs,
    making training and test different choice processes);
  * the attention half-life is LOCKED on the OTC training likelihood
    (14 d, the training-optimal value) before any test-period evaluation;
    the grid {3, 7, 30} d is reported sensitivity only;
  * Alpha re-estimates coefficients only under the locked specification
    (same models, same log1p(H) transform, same risk-set rules, same
    metrics); no half-life selection on Alpha;
  * confirmation criteria fixed before the Alpha analysis: ll_Alpha(M2)
    > ll_Alpha(M1) and ll_Alpha(M3) - ll_Alpha(M2) near zero or negative,
    with moving-block bootstrap intervals on the per-event log-score
    differences;
  * bounded-response variants M2b/M3b use
    psi_H(H) = exp{g tanh(b H)} (bounded, Lipschitz for fixed g): the
    feature is tanh(b H) and the exponent g is estimated freely by the
    conditional logit.  The shape b is selected on the OTC training
    likelihood from 8,000 complete strata subsampled out of the
    FULL-history case-control (H built from the complete training stream;
    strata are the sampling unit, never individual rows -- sampling
    events first would thin the stream and change H, the risk set and the
    used-pair history) and LOCKED for Alpha, so the empirical
    specification matches the class of response functions the planned
    theorem assumes.

Design notes.
  * g(t) is a stratum-level constant in the case-control likelihood (every
    candidate in one stratum faces the same t), so it is NOT identifiable
    from target choice; it is evaluated only at the macro (arrival-rate)
    level, as a training-end rate extrapolation vs the static rate.
  * H_j(t) = sum over incoming events exp(-rho (t - tau_m)), i.e.
    dH = -rho H dt + dN_in, implemented by AttentionEngine.
  * All covariates at test time are built from state replayed through the
    training stream to the cutoff, then evolved event by event: capitals,
    attention, indegree, recency, sent-history, used pairs.

Output: experiments/ablation_attention.json (self-contained).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dengyunetwork import estimation as EST  # noqa: E402
from dengyunetwork.data import (  # noqa: E402
    TimeSplit, load_events, node_entry_times, sort_by_time,
)
from dengyunetwork.facts import (  # noqa: E402
    AttentionEngine, CapitalEngine, ETA, LN2,
)

EXP_DIR = ROOT / "experiments"
K_GROUPS = 5
HL_CAP = 180.0          # capital half-life (days), fixed at the baseline
HL_PRIMARY = 14.0          # attention half-life locked on OTC training llf
HL_GRID = [3.0, 7.0, 30.0]  # attention half-life sensitivity (reported only)
N_CONTROLS = 30
TOPK = 10


def load_fitted():
    """Load the WP5 estimation cache (fit_schema 6)."""
    fj = json.loads((EXP_DIR / "fitted_params.json").read_text())
    if fj.get("fit_schema") != 6:
        raise RuntimeError("fitted_params.json is not fit_schema 6; "
                           "rerun scripts/06_holdout.py first")
    a_hat = pd.Series({int(k): v for k, v in fj["a_hat"].items()}).astype(float)
    groups = pd.Series({int(k): int(v) for k, v in fj["groups"].items()})
    W_hat = np.array(fj["W_hat"])
    delta = fj["delta"]
    return a_hat, groups, W_hat, delta


def fit_nested(cc: pd.DataFrame, hl_h: float, bounded: bool = False,
               double_bounded: bool = False) -> dict:
    """Fit M0..M3 (and the bounded-response variants M2b/M3b, and the
    double-bounded M3bb with a bounded capital channel tanhC) on the
    training case-control data; returns coefficients and training llf.

    bounded=True uses tanhH (bounded, Lipschitz attention response
    psi_H(H) = exp{g tanh(b H)}, g free) in place of logH for the H
    channel.  double_bounded=True further uses tanhC (bounded capital
    response exp{g tanh(b C)} on net capital) in place of the unbounded
    linear net_cap/count_cap terms, so the empirical specification
    matches the class of response functions the planned theorem assumes
    on BOTH channels.
    """
    if double_bounded:
        h_feat, c_feats = "tanhH", ["tanhC"]
    elif bounded:
        h_feat, c_feats = "tanhH", ["net_cap", "count_cap"]
    else:
        h_feat, c_feats = "logH", ["net_cap", "count_cap"]
    feats_m = {
        "M0": ["logW"],
        "M1": ["logW"] + c_feats,
        "M2": ["logW", h_feat],
        "M3": ["logW"] + c_feats + [h_feat],
    }
    tag = "bb" if double_bounded else ("b" if bounded else "")
    out = {}
    for name, feats in feats_m.items():
        res = EST.conditional_logit_fit(cc, feats)
        out[name] = {"feats": feats,
                     "coef": {f: float(res.params[f]) for f in feats},
                     "train_llf": float(res.llf)}
        print(f"  [{name}, hlH={hl_h:g}d{tag}] "
              f"llf={res.llf:.1f}  "
              f"coef={ {k: round(v, 4) for k, v in out[name]['coef'].items()} }")
    return out


def select_psi_h(cc: pd.DataFrame, beta_grid) -> dict:
    """Select the bounded attention response shape beta on the training
    likelihood only (training case-control; no test information).

    The response is psi_H(H) = exp{g tanh(b H)}: the feature is
    tanh(b H) and the exponent g is estimated freely by the conditional
    logit, so the effective response is bounded (range [e^{-|g|}, e^{|g|}]
    for fixed g) and Lipschitz.  Only b is grid-searched; g is free.
    H is recovered from the logH column (H = expm1(logH)) so no extra
    case-control pass is needed per grid point.
    """
    H = np.expm1(cc["logH"].values)
    best = None
    for b in beta_grid:
        cc_ = cc.copy()
        cc_["tanhH"] = np.tanh(b * H)
        res = EST.conditional_logit_fit(cc_, ["logW", "tanhH"])
        llf = float(res.llf)
        if best is None or llf > best["llf"]:
            best = {"beta": float(b), "llf": llf}
    print(f"  [psi_H shape] selected beta={best['beta']} "
          f"(train llf={best['llf']:.1f})")
    return best


def evaluate_attention(test, train, nodes, pos, groups, W_hat, delta,
                       models, rem_beta, rho_h, psi_h=None, psi_c=None):
    """Out-of-sample test-period metrics for the nested models.

    Models maps model name -> {"feats": [...], "coef": {...}}; the features
    are evaluated per candidate at each test event.  rho_h is the attention
    decay used to construct logH, matching the half-life the models were
    fitted with (strictly out-of-sample: the test H state is replayed from
    the training stream through the cutoff, never re-estimated).  psi_h,
    when given, adds the bounded-response feature tanhH = tanh(beta H)
    (response exp{g tanh(beta H)}, g free) used by the M2b/M3b variants.
    psi_c, when given, adds the bounded capital feature tanhC =
    tanh(beta_c * net_cap) used by the double-bounded M3bb variant.
    The REM baseline is scored from its cached training coefficients as a
    reference.
    """
    test_nodes = np.unique(np.concatenate(
        [test["source"].values, test["target"].values]))
    extra = np.setdiff1d(test_nodes, nodes)
    if len(extra):
        nodes = np.concatenate([nodes, extra])
        pos = {n: k for k, n in enumerate(nodes)}
        g_arr = np.concatenate([groups.values,
                                -np.ones(len(extra), dtype=int)])
    else:
        g_arr = groups.values
    N = len(nodes)
    eng = CapitalEngine(N, delta)
    eng_cnt = CapitalEngine(N, delta)
    att = AttentionEngine(N, rho=rho_h)
    indeg = np.zeros(N, dtype=int)
    sent = np.zeros(N, dtype=int)
    last_recv = np.full(N, -np.inf)
    active = np.zeros(N, dtype=bool)
    used: dict[int, set[int]] = {}
    for row in train.itertuples(index=False):
        pi_tr, pj_tr = pos[row.source], pos[row.target]
        used.setdefault(pi_tr, set()).add(pj_tr)
        active[pi_tr] = active[pj_tr] = True
        indeg[pj_tr] += 1
        sent[pi_tr] += 1
        last_recv[pj_tr] = row.timestamp
        eng.jump(pj_tr, row.timestamp, ETA * row.rating)
        eng_cnt.jump(pj_tr, row.timestamp, 1.0)
        att.jump(pj_tr, row.timestamp)
    entry_t = np.full(N, np.inf)
    for nd, ent in node_entry_times(
            pd.concat([train, test]))["entry"].items():
        if nd in pos:
            entry_t[pos[nd]] = ent
    rng = np.random.default_rng(0)
    n_test = len(test)
    names = list(models) + ["rem"]
    scores = {m: np.zeros(n_test) for m in names}
    ranks = {m: np.zeros(n_test) for m in models}
    rem_feats = ["log_indegree", "net_cap", "count_cap", "age_days",
                 "log_recency", "ever_sent", "logW"]

    def feats_at(cand, t):
        net_at = eng.at_all(t)[cand]
        cnt_at = eng_cnt.at_all(t)[cand]
        h_at = att.at_all(t)[cand]
        age = (t - entry_t[cand]) / 86400.0
        rec_raw = (t - last_recv[cand]) / 86400.0
        rec = np.where(np.isfinite(rec_raw), rec_raw, age)
        gi = g_arr[pi_] if g_arr[pi_] >= 0 else -1
        gc = g_arr[cand]
        valid = (gi >= 0) & (gc >= 0)
        logW = np.zeros(len(cand))
        logW[valid] = np.log(W_hat[gi, gc[valid]] + 1e-9)
        feats = {"net_cap": net_at, "count_cap": cnt_at, "logH": np.log1p(h_at),
                 "log_indegree": np.log1p(indeg[cand]), "age_days": age,
                 "log_recency": np.log1p(np.maximum(rec, 0.0)),
                 "ever_sent": (sent[cand] > 0).astype(float), "logW": logW}
        if psi_h is not None:
            b_h = psi_h
            feats["tanhH"] = np.tanh(b_h * h_at)
        if psi_c is not None:
            feats["tanhC"] = np.tanh(psi_c * net_at)
        return feats

    for e, row in enumerate(test.itertuples(index=False)):
        i, j, r, t = row.source, row.target, row.rating, row.timestamp
        pi_, pj = pos[i], pos[j]
        active[pi_] = active[pj] = True
        act = np.flatnonzero(active)
        # legal risk set, identical to the training-period construction:
        # exclude the sender and already-rated ordered pairs
        others = act[(act != pj) & (act != pi_)]
        avail = np.array([c for c in others
                          if c not in used.get(pi_, set())])
        n_ctrl = min(N_CONTROLS, len(avail))
        ctrl = rng.choice(avail, size=n_ctrl, replace=False)
        cand = np.concatenate([[pj], ctrl]).astype(np.int64)
        X = feats_at(cand, t)
        for m in names:
            if m == "rem":
                sc = sum(rem_beta[f] * X[f] for f in rem_feats)
            else:
                sc = sum(models[m]["coef"].get(f, 0.0) * X[f]
                         for f in models[m]["feats"])
            z = sc - sc.max()
            scores[m][e] = z[0] - np.log(np.exp(z).sum())
        # full-risk-set ranking (all active, not-yet-used, excluding sender)
        avail_all = np.array([c for c in act
                              if c != pi_
                              and c not in used.get(pi_, set())])
        pos_j = int(np.flatnonzero(avail_all == pj)[0])
        Xa = feats_at(avail_all, t)
        for m in models:
            sc = sum(models[m]["coef"].get(f, 0.0) * Xa[f]
                     for f in models[m]["feats"])
            order = np.argsort(-sc)
            ranks[m][e] = int(np.flatnonzero(order == pos_j)[0]) + 1
        # realised-event state update
        used.setdefault(pi_, set()).add(pj)
        indeg[pj] += 1
        sent[pi_] += 1
        eng_cnt.jump(pj, t, 1.0)
        eng.jump(pj, t, ETA * r)
        att.jump(pj, t)
        last_recv[pj] = t

    out = {m: {"partial_loglik": float(scores[m].mean())}
           for m in names}
    for m in models:
        out[m]["top10_recall"] = float((ranks[m] <= TOPK).mean())
        out[m]["mrr"] = float((1.0 / ranks[m]).mean())
    out["per_event"] = {"M0": scores["M0"].tolist(), "M1": scores["M1"].tolist(),
                        "M2": scores["M2"].tolist(), "M3": scores["M3"].tolist()}
    return out


def paired_block_bootstrap(scores: dict, n_boot: int = 1000, block: int = 25,
                           seed: int = 0, label: str = "",
                           pairs=None) -> dict:
    """Paired moving-block bootstrap on per-event log-scores.

    For each pair of models (default: M2-M1, M3-M2) computes the per-event
    score difference; the bootstrap resamples contiguous blocks of 25
    consecutive test events and reports the 95% percentile interval of
    the mean difference.  This is the prespecified uncertainty statement
    for ``H beats C'' and ``C adds nothing on top of H''.
    """
    rng = np.random.default_rng(seed)
    n = len(scores["M0"])
    n_blocks = n // block
    if pairs is None:
        pairs = [("M2", "M1"), ("M3", "M2")]
    out = {}
    for a, b in pairs:
        d = np.array(scores[a]) - np.array(scores[b])
        mean_d = float(d.mean())
        boot = np.zeros(n_boot)
        for k in range(n_boot):
            idx = []
            for _ in range(n_blocks):
                start = rng.integers(0, n - block + 1)
                idx.extend(range(start, start + block))
            idx = np.array(idx[:n])
            boot[k] = d[idx].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[f"d{a}{b}{label}"] = {"mean": mean_d, "ci95": [float(lo), float(hi)],
                                  "covers_zero": bool(lo <= 0 <= hi),
                                  "n_boot": n_boot,
                                  "block": block,
                                  "note": ("moving-block bootstrap with "
                                           f"blocks of {block} consecutive "
                                           "events")}
    return out


def calendar_block_bootstrap(scores: dict, times: np.ndarray, days: float,
                             n_boot: int = 1000, seed: int = 0,
                             label: str = "") -> dict:
    """Calendar-time block bootstrap on per-event log-scores.

    Blocks are contiguous windows of ``days`` calendar days; each
    resample draws blocks with replacement.  Because attention memory has
    a 14-day half-life, the calendar block length matches the dependence
    scale of H directly (the fixed event-count blocks do not, and event
    density differs across platforms).
    """
    rng = np.random.default_rng(seed)
    tmin, tmax = float(times.min()), float(times.max())
    nb = max(int(np.ceil((tmax - tmin) / (days * 86400.0))), 1)
    edges = np.linspace(tmin, tmax, nb + 1)
    bidx = np.clip(np.digitize(times, edges[1:-1]), 0, nb - 1)
    blocks = [np.flatnonzero(bidx == k) for k in range(nb)]
    n = len(scores["M0"])
    pairs = [("M2", "M1"), ("M3", "M2")]
    out = {}
    for a, b in pairs:
        d = np.array(scores[a]) - np.array(scores[b])
        mean_d = float(d.mean())
        boot = np.zeros(n_boot)
        for k in range(n_boot):
            idx = np.concatenate([blocks[j] for j in rng.integers(0, nb, nb)])
            idx = idx[:n]
            boot[k] = d[idx].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[f"d{a}{b}{label}"] = {"mean": mean_d, "ci95": [float(lo), float(hi)],
                                  "covers_zero": bool(lo <= 0 <= hi),
                                  "n_boot": n_boot, "block_days": days,
                                  "n_blocks": nb,
                                  "note": ("calendar-time block bootstrap "
                                           f"with {days:g}-day blocks")}
    return out


def block_sensitivity(scores: dict, times: np.ndarray,
                      event_blocks=(25, 50, 100),
                      day_blocks=(14.0, 28.0, 56.0),
                      n_boot: int = 1000) -> dict:
    """Block-length sensitivity for the paired differences.

    The M3-M2 increment is small, so its uncertainty must be checked
    against the block length (event-count and calendar-time variants).
    """
    out = {}
    for blk in event_blocks:
        out[f"event_block_{blk}"] = paired_block_bootstrap(
            scores, n_boot=n_boot, block=blk, seed=0)
    for d_ in day_blocks:
        out[f"calendar_block_{d_:g}d"] = calendar_block_bootstrap(
            scores, times, d_, n_boot=n_boot, seed=0)
    return out


def macro_g(train, test):
    """Macro-level check of the platform time effect g(t).

    Predicts the test-period per-capita event rate from (a) the static
    training-average rate and (b) a training-end (last 30 d) extrapolation,
    and reports the IAE of each against the realised test rate.  This is
    deliberately minimal: it isolates whether a time-varying rate scale
    fixes the arrival-rate failure without re-solving the full system.
    """
    t0 = train["timestamp"].min()
    t1 = train["timestamp"].max()
    n_nodes = len(set(train["source"]) | set(train["target"]))
    span_d = (t1 - t0) / 86400.0
    rate_static = len(train) / span_d / n_nodes
    t_end = t1 - 30 * 86400.0
    last30 = train[train["timestamp"] >= t_end]
    span30 = (t1 - t_end) / 86400.0
    rate_end = (len(last30) / span30 / n_nodes) if len(last30) else rate_static
    # realised test rate per 30-day bin (events per person-day)
    tmin, tmax = test["timestamp"].min(), test["timestamp"].max()
    nb = max(int(np.ceil((tmax - tmin) / (30 * 86400.0))), 1)
    bins = np.linspace(tmin, tmax, nb + 1)
    real, _ = np.histogram(test["timestamp"], bins=bins)
    real_rate = real / ((tmax - tmin) / nb / 86400.0) / n_nodes
    n_bin = len(real_rate)
    iae_static = float(np.abs(real_rate - rate_static).mean()
                       / max(real_rate.mean(), 1e-12))
    iae_end = float(np.abs(real_rate - rate_end).mean()
                    / max(real_rate.mean(), 1e-12))
    return {"rate_static": float(rate_static), "rate_train_end": float(rate_end),
            "realised_mean": float(real_rate.mean()),
            "iae_static_rel": iae_static, "iae_train_end_rel": iae_end,
            "n_30d_bins": int(n_bin)}


def run_ablation(dataset: str, hl_primary: float, hl_sensitivity: list,
                 exclude_used: bool = True, n_boot: int = 1000,
                 psi_beta: float | None = None,
                 psi_c_beta: float | None = None) -> dict:
    """Full ablation for one dataset.

    Pre-registered protocol (user review, 2026-08-12):
      * the attention half-life is selected on the OTC training likelihood
        (14 d) and locked before any test-period evaluation; the grid
        {3, 7, 30} d is a reported sensitivity, never a test-period choice;
      * the legal risk set (active, not the sender, not an already-rated
        ordered pair) is used identically in the training case-control
        construction and in the test-period evaluation;
      * Alpha re-estimates the model coefficients only: same M0-M3
        definitions, same log1p(H) transform, same risk-set rules, same
        metrics; no half-life selection on Alpha.
    """
    t_start = time.time()
    if dataset == "otc":
        a_hat, groups, W_hat, delta = load_fitted()
        df = sort_by_time(load_events())
        entry = node_entry_times(df)
    elif dataset == "alpha":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rob07", ROOT / "scripts" / "07_robustness.py")
        rob07 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rob07)
        alpha_path = ROOT / "data" / "bitcoin_alpha" / \
            "soc-sign-bitcoinalpha.csv.gz"
        df = sort_by_time(pd.read_csv(
            alpha_path, names=["source", "target", "rating", "timestamp"],
            compression="gzip"))
        entry = node_entry_times(df)
        delta = LN2 / (HL_CAP * 86400.0)
        split_a = TimeSplit.by_fraction_of_span(df, 0.7)
        # full estimation on Alpha's own training period (psi profile
        # included; nothing is pooled from OTC)
        fit_a = rob07.fit_pipeline(split_a.train, entry, delta, HL_CAP,
                                   "alpha", psi_profile=True)
        a_hat, groups, W_hat = fit_a["a_hat"], fit_a["groups"], fit_a["W_hat"]
    else:
        raise ValueError(dataset)
    split = TimeSplit.by_fraction_of_span(df, 0.7)
    train, test = split.train, split.test
    print(f"[{dataset}] train {len(train)} / test {len(test)} events; "
          f"primary attention half-life {hl_primary} d, sensitivity "
          f"{hl_sensitivity} d")

    # ---- REM reference re-fitted on the legal risk set ----
    # (the cached 06 coefficients were fit on the old, wider risk set;
    #  for a clean nested comparison the REM row must share the same
    #  legal-risk-set case-control as M0-M3)
    cc_rem = EST.build_case_control(train, groups, entry, delta,
                                    a_hat=a_hat, W_hat=W_hat,
                                    n_controls=N_CONTROLS, seed=0,
                                    exclude_used=exclude_used)
    rem_feats = ["log_indegree", "net_cap", "count_cap", "age_days",
                 "log_recency", "ever_sent", "logW"]
    rem_res = EST.conditional_logit_fit(cc_rem, rem_feats)
    rem_beta = {f: float(rem_res.params[f]) for f in rem_feats}
    print(f"  REM (legal risk set): coef = "
          f"{ {k: round(v, 3) for k, v in rem_beta.items()} }")

    # ---- training-period case-control per attention half-life ----
    rho_h = {h: LN2 / (h * 86400.0) for h in [hl_primary] + hl_sensitivity}
    cc_by_hl = {}
    for h in [hl_primary] + hl_sensitivity:
        t0 = time.time()
        cc = EST.build_case_control(train, groups, entry, delta,
                                    a_hat=a_hat, W_hat=W_hat,
                                    n_controls=N_CONTROLS, seed=0,
                                    h_rho=rho_h[h],
                                    exclude_used=exclude_used)
        cc_by_hl[h] = cc
        print(f"  case-control hlH={h:g}d: {len(cc)} rows "
              f"({time.time()-t0:.0f}s)")

    # ---- nested fits (unbounded logH channel) ----
    models_by_hl = {}
    for h in [hl_primary] + hl_sensitivity:
        print(f"  fitting nested models, attention half-life {h} d")
        models_by_hl[h] = fit_nested(cc_by_hl[h], h)

    # ---- bounded attention response psi_H(H) = exp{g tanh(b H)} ----
    # shape b is selected on the OTC training likelihood only (no test
    # information) and LOCKED for Alpha (--psi-beta 0.5): the confirmatory
    # dataset re-estimates coefficients only, never the shape.  The OTC
    # grid search fits a random time subsample of 8,000 events (NOT the
    # earliest 6,000: the platform early phase has almost no H history,
    # which would bias the shape selection; the psi2 profile in 06 had
    # the same head() issue, noted here as a deliberate deviation).
    if psi_beta is None:
        t0 = time.time()
        # IMPORTANT (user review, 2026-08-12): the shape must be selected on
        # H built from the COMPLETE training history.  Sampling events first
        # and then building case-control would thin the event stream and
        # change H_j(t) = sum exp(-rho (t - tau_m)), the risk set, and the
        # used-pair history.  So we build the full case-control with the
        # full training stream, then subsample COMPLETE STRATA (never
        # individual rows).
        cc_sel_full = EST.build_case_control(train, groups, entry, delta,
                                             a_hat=a_hat, W_hat=W_hat,
                                             n_controls=N_CONTROLS, seed=0,
                                             h_rho=rho_h[hl_primary],
                                             exclude_used=exclude_used)
        strata_all = cc_sel_full["stratum"].unique()
        strata_sel = np.random.default_rng(0).choice(
            strata_all, size=min(8000, len(strata_all)), replace=False)
        cc_sel = cc_sel_full[cc_sel_full["stratum"].isin(strata_sel)]
        psi_shape = select_psi_h(cc_sel,
                                 beta_grid=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
        print(f"  psi_H shape selection wall time {time.time()-t0:.0f}s")
        psi_h = psi_shape["beta"]
    else:
        # locked shape carried over from OTC; do not re-select on Alpha
        psi_shape = {"beta": psi_beta, "llf": None,
                     "note": "locked from OTC (prespecified)"}
        psi_h = psi_beta
        print(f"  psi_H shape LOCKED from OTC: beta={psi_h} (no re-selection)")
    # bounded case-control (primary half-life) for the M2b/M3b fits
    cc_b = EST.build_case_control(train, groups, entry, delta,
                                  a_hat=a_hat, W_hat=W_hat,
                                  n_controls=N_CONTROLS, seed=0,
                                  h_rho=rho_h[hl_primary],
                                  exclude_used=exclude_used,
                                  psi_exp_beta=psi_h)
    print(f"  fitting bounded-response models (hlH={hl_primary:g}d)")
    models_b = {hl_primary: fit_nested(cc_b, hl_primary, bounded=True)}
    # sensitivity grid for the bounded channel uses the same locked shape
    for h in hl_sensitivity:
        cc_bh = EST.build_case_control(train, groups, entry, delta,
                                       a_hat=a_hat, W_hat=W_hat,
                                       n_controls=N_CONTROLS, seed=0,
                                       h_rho=rho_h[h],
                                       exclude_used=exclude_used,
                                       psi_exp_beta=psi_h)
        print(f"  fitting bounded-response models (hlH={h:g}d)")
        models_b[h] = fit_nested(cc_bh, h, bounded=True)

    # ---- double-bounded capital channel psi_C(C) = exp{g tanh(b C)} ----
    # shape b_c selected on the OTC training likelihood (full-history
    # strata, same protocol as b_h) and locked for Alpha; only used by the
    # M3bb variant, so the capital channel also matches the theorem's
    # bounded-Lipschitz class.  Runs AFTER psi_h is resolved.
    if psi_c_beta is None:
        # build the selection case-control independently (locked psi_beta
        # on Alpha does not define cc_sel)
        cc_csel = EST.build_case_control(train, groups, entry, delta,
                                         a_hat=a_hat, W_hat=W_hat,
                                         n_controls=N_CONTROLS, seed=0,
                                         h_rho=rho_h[hl_primary],
                                         exclude_used=exclude_used)
        strata_all_c = cc_csel["stratum"].unique()
        strata_sel_c = np.random.default_rng(1).choice(
            strata_all_c, size=min(8000, len(strata_all_c)), replace=False)
        cc_csel = cc_csel[cc_csel["stratum"].isin(strata_sel_c)]
        Hsel = np.expm1(cc_csel["logH"].values)
        best_c = None
        for bc in [0.1, 0.25, 0.5, 1.0]:
            cc_c = cc_csel.copy()
            cc_c["tanhC"] = np.tanh(bc * cc_c["net_cap"].values)
            cc_c["tanhH"] = np.tanh(psi_h * Hsel)
            res = EST.conditional_logit_fit(
                cc_c, ["logW", "tanhC", "tanhH"])
            llf = float(res.llf)
            if best_c is None or llf > best_c["llf"]:
                best_c = {"beta": float(bc), "llf": llf}
        psi_c = best_c["beta"]
        print(f"  [psi_C shape] selected beta={psi_c} "
              f"(train llf={best_c['llf']:.1f})")
    else:
        psi_c = psi_c_beta
        print(f"  psi_C shape LOCKED from OTC: beta={psi_c}")
    cc_bb = EST.build_case_control(train, groups, entry, delta,
                                   a_hat=a_hat, W_hat=W_hat,
                                   n_controls=N_CONTROLS, seed=0,
                                   h_rho=rho_h[hl_primary],
                                   exclude_used=exclude_used,
                                   psi_exp_beta=psi_h, psi_c_beta=psi_c)
    print(f"  fitting double-bounded models (hlH={hl_primary:g}d)")
    models_bb = {hl_primary: fit_nested(cc_bb, hl_primary,
                                        double_bounded=True)}

    # ---- out-of-sample evaluation ----
    nodes_all = np.asarray(groups.index)
    pos_all = {n: k for k, n in enumerate(nodes_all)}
    for h in [hl_primary] + hl_sensitivity:
        print(f"  test-period evaluation, attention half-life {h} d")
        models_by_hl[h]["test"] = evaluate_attention(
            test, train, nodes_all, pos_all, groups, W_hat, delta,
            models_by_hl[h], rem_beta, rho_h[h])
        models_b[h]["test"] = evaluate_attention(
            test, train, nodes_all, pos_all, groups, W_hat, delta,
            models_b[h], rem_beta, rho_h[h], psi_h=psi_h)
    # double-bounded M3bb evaluated at the primary half-life only
    models_bb[hl_primary]["test"] = evaluate_attention(
        test, train, nodes_all, pos_all, groups, W_hat, delta,
        models_bb[hl_primary], rem_beta, rho_h[hl_primary],
        psi_h=psi_h, psi_c=psi_c)

    # ---- paired block bootstrap on the primary half-life ----
    primary = models_by_hl[hl_primary]
    primary_b = models_b[hl_primary]
    boot = paired_block_bootstrap(primary["test"]["per_event"],
                                  n_boot=n_boot, block=25, seed=0)
    boot_b = paired_block_bootstrap(primary_b["test"]["per_event"],
                                    n_boot=n_boot, block=25, seed=1,
                                    label="_b")
    # block-length sensitivity (event-count and calendar-time blocks)
    test_times = test["timestamp"].values
    boot_sens = block_sensitivity(primary["test"]["per_event"], test_times,
                                  n_boot=n_boot)
    boot_sens_b = block_sensitivity(primary_b["test"]["per_event"], test_times,
                                    n_boot=n_boot)
    # double-bounded M3bb vs bounded M2b: per-event difference needs
    # scores from both models, so compute directly (M3bb - M2b)
    per_b = primary_b["test"]["per_event"]
    per_bb = models_bb[hl_primary]["test"]["per_event"]
    d_bb = (np.array(per_bb["M3"]) - np.array(per_b["M2"]))
    mean_bb = float(d_bb.mean())
    rng_bb = np.random.default_rng(2)
    n_ev = len(d_bb)
    n_blk = n_ev // 25
    boot_arr = np.zeros(n_boot)
    for k in range(n_boot):
        idx = []
        for _ in range(n_blk):
            s0 = rng_bb.integers(0, n_ev - 25 + 1)
            idx.extend(range(s0, s0 + 25))
        boot_arr[k] = d_bb[np.array(idx[:n_ev])].mean()
    lo, hi = np.percentile(boot_arr, [2.5, 97.5])
    boot_bb = {"dM3bbM2b": {"mean": mean_bb, "ci95": [float(lo), float(hi)],
                            "covers_zero": bool(lo <= 0 <= hi),
                            "n_boot": n_boot, "block": 25,
                            "note": ("moving-block bootstrap on the "
                                     "double-bounded M3bb minus bounded "
                                     "M2b per-event log scores")}}

    # ---- macro g(t) ----
    macro = macro_g(train, test)

    out = {
        "dataset": dataset,
        "split": {"train": len(train), "test": len(test),
                  "cutoff_unix": split.cutoff},
        "hl_primary_days": hl_primary,
        "hl_sensitivity_days": hl_sensitivity,
        "risk_set": {"exclude_used_pairs": exclude_used,
                     "exclude_sender": True,
                     "note": ("legal risk set used identically in "
                              "training and test")},
        "psi_h": {"beta": psi_h,
                  "train_llf": psi_shape.get("llf"),
                  "form": "exp{g tanh(b H)}, g free (bounded, Lipschitz)",
                  "note": (psi_shape.get("note")
                           or ("shape b selected on 8,000 complete strata "
                               "subsampled from the FULL-history OTC "
                               "case-control at the primary half-life "
                               "(H built from the complete training "
                               "stream; strata are the sampling unit, "
                               "never individual rows); locked for "
                               "sensitivity and Alpha"))},
        "nested": {f"hlH={h}d": {k: v for k, v in models_by_hl[h].items()
                                 if k != "test"}
                   for h in [hl_primary] + hl_sensitivity},
        "nested_bounded": {f"hlH={h}d": {k: v
                                         for k, v in models_b[h].items()
                                         if k != "test"}
                           for h in [hl_primary] + hl_sensitivity},
        "test": {f"hlH={h}d": {k: v for k, v in models_by_hl[h]["test"].items()
                               if k != "per_event"}
                 for h in [hl_primary] + hl_sensitivity},
        "test_bounded": {f"hlH={h}d": {k: v
                                       for k, v in models_b[h]["test"].items()
                                       if k != "per_event"}
                         for h in [hl_primary] + hl_sensitivity},
        "psi_c": {"beta": psi_c,
                  "form": "exp{g tanh(b C)} on net capital, g free",
                  "note": (("selected on OTC full-history strata "
                            "(8,000 complete strata, seed 1); "
                            "locked for Alpha")
                           if psi_c_beta is None
                           else "locked from OTC (prespecified)")},
        "nested_double_bounded": {k: v
                                  for k, v in
                                  models_bb[hl_primary].items()
                                  if k != "test"},
        "test_double_bounded": {k: v
                                for k, v in
                                models_bb[hl_primary]["test"].items()
                                if k != "per_event"},
        "paired_bootstrap": boot,
        "paired_bootstrap_bounded": boot_b,
        "paired_bootstrap_double_bounded": boot_bb,
        "block_sensitivity": boot_sens,
        "block_sensitivity_bounded": boot_sens_b,
        "macro_g": macro,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["otc", "alpha"], default="otc")
    p.add_argument("--hl-primary", type=float, default=14.0,
                   help="attention half-life locked on OTC training "
                        "likelihood (14 d); the confirmatory value")
    p.add_argument("--hl-sensitivity", nargs="*", type=float,
                   default=[3.0, 7.0, 30.0],
                   help="sensitivity grid (reported, not selected)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--psi-beta", type=float, default=None,
                   help="bounded-response shape b; None = select on this "
                        "dataset's training data, float = lock it "
                        "(Alpha must pass the OTC-selected value)")
    p.add_argument("--psi-c-beta", type=float, default=None,
                   help="bounded capital shape b_c for M3bb; None = select "
                        "on OTC training data, float = lock it")
    args = p.parse_args()

    EXP_DIR.mkdir(exist_ok=True)
    out = run_ablation(args.dataset, args.hl_primary,
                       args.hl_sensitivity, n_boot=args.n_boot,
                       psi_beta=args.psi_beta,
                       psi_c_beta=args.psi_c_beta)
    fp = EXP_DIR / f"ablation_attention_{args.dataset}.json"
    fp.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"test": out["test"], "paired_bootstrap": out["paired_bootstrap"],
                      "macro_g": out["macro_g"]}, indent=2))
    print(f"\n[ok] wrote {fp.name} ({out['wall_seconds']}s)")


if __name__ == "__main__":
    main()
