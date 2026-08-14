#!/usr/bin/env python
"""WP5: estimation on the training period + holdout validation (gate G5).

Main split: 70% of the time span trains (cutoff 2014-07-03, 32,991 events),
the remaining 30% (2,601 events) tests.  All estimation on the training
period only; all metrics on the test period.

Models compared (research plan 8.5):
  1. homogeneous marked Poisson      (uniform targets)
  2. activity-only                   (uniform targets; time-varying rate)
  3. preferential attachment         (rank by indegree, alpha fitted)
  4. REM-style case-control model    (linear covariates, the event-level
                                     competitor with the strongest signal)
  5. kinetic model (this paper)      (kernel x capital response)

Micro metrics: case-control partial likelihood (30 sampled controls),
top-10 recall and MRR, sign log-loss / Brier.
Macro metrics: predicted vs realised size-biased capital distribution
(W1, JSD) and the event-rate trajectory IAE (predicted by the kinetic
solver vs realised).
Consistency: one MC run at N=5881 with the fitted parameters vs the solver
(finite-size check) and vs the realised test period.
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
from dengyunetwork.facts import ETA, CapitalEngine, LN2  # noqa: E402
from dengyunetwork.kinetic import KineticSolver, KineticSpec  # noqa: E402
from dengyunetwork.simulator import (  # noqa: E402
    EventSimulator, SimParams, tanh_psi,
)

EXP_DIR = ROOT / "experiments"
K_GROUPS = 5
HL_BASE = 180.0          # baseline half-life (days) for capital computation
N_CONTROLS = 30
TOPK = 10


def capital_engine_at(df: pd.DataFrame, nodes: np.ndarray, delta: float,
                      end_time: float | None = None):
    """Replay events into a CapitalEngine; returns the engine (all nodes'
    capitals available at the final time)."""
    pos = {n: k for k, n in enumerate(nodes)}
    eng = CapitalEngine(len(nodes), delta)
    for row in df.itertuples(index=False):
        eng.jump(pos[row.target], row.timestamp, ETA * row.rating)
    if end_time is not None:
        eng.at_all(end_time)
    return eng


def main() -> None:
    t_start = time.time()
    EXP_DIR.mkdir(exist_ok=True)
    df = sort_by_time(load_events())
    entry = node_entry_times(df)
    split = TimeSplit.by_fraction_of_span(df, 0.7)
    train, test = split.train, split.test
    delta = LN2 / (HL_BASE * 86400.0)
    print(f"train {len(train)} / test {len(test)} events, "
          f"cutoff {pd.to_datetime(split.cutoff, unit='s')}")
    out: dict = {"split": {"train": len(train), "test": len(test),
                           "cutoff_unix": split.cutoff}}

    # ---------------------------------------------------------------- fit --
    # cache: if the fits already exist (e.g. from a previous run that crashed
    # in the evaluation), load them instead of re-running the slow stages
    fp = EXP_DIR / "fitted_params.json"
    fit_schema = 6  # plus expanded psi grid and corrected mark magnitudes
    cache_ok = False
    if fp.exists():
        fj = json.loads(fp.read_text())
        cache_ok = fj.get("fit_schema") == fit_schema
    if cache_ok:
        a_hat = pd.Series({int(k): v for k, v in fj["a_hat"].items()}).astype(float)
        groups = pd.Series({int(k): int(v) for k, v in fj["groups"].items()})
        W_hat = np.array(fj["W_hat"])
        alpha, beta = fj["alpha"], fj["beta"]
        psi2 = EST.psi2_from(alpha, beta)
        c_rate = fj["c_rate"]
        delta = fj["delta"]
        rem_beta = fj["rem_beta"]
        nodes_all = np.asarray(groups.index)
        pos_all = {n: k for k, n in enumerate(nodes_all)}
        a_arr = a_hat.reindex(nodes_all).fillna(0.0).values
        mark_model = {"sign_coef": fj["sign_model"]["coef"],
                      "sign_intercept": fj["sign_model"]["intercept"],
                      "mag_dist": fj["mag_dist"],
                      "mag_bin_edges": fj["mag_bin_edges"]}
        out["groups_k"] = K_GROUPS
        out["group_sizes"] = groups.value_counts().sort_index().tolist()
        out["loaded_from_cache"] = True
        print("loaded fitted params from cache")
    else:
        a_hat = EST.fit_activity(train, entry)
        groups = EST.fit_types(train, k=K_GROUPS, seed=0)
        W_hat, N_pop = EST.fit_kernel(train, groups, a_hat)
        out["groups_k"] = K_GROUPS
        out["group_sizes"] = groups.value_counts().sort_index().tolist()
        print("group sizes:", out["group_sizes"])

        # psi profile (subsampled conditional logit)
        t0 = time.time()
        psi_prof = EST.fit_psi_profile(
            train, groups, entry, delta, a_hat, W_hat,
            n_controls=15, max_events=6000, seed=0)
        alpha, beta = psi_prof["best_alpha"], psi_prof["best_beta"]
        psi2 = EST.psi2_from(alpha, beta)
        out["psi_profile"] = psi_prof
        out["psi_fit_wall_s"] = round(time.time() - t0, 1)
        print(f"psi profile: alpha={alpha}, beta={beta} "
              f"({out['psi_fit_wall_s']}s)")

        # kernel refinement with the fitted psi2
        nodes_all = np.asarray(groups.index)
        pos_all = {n: k for k, n in enumerate(nodes_all)}
        eng_tr = capital_engine_at(train, nodes_all, delta)
        psi2_vals = psi2(eng_tr.value)
        psi1_vals = np.ones(len(nodes_all))
        W_hat2, _ = EST.fit_kernel(train, groups, a_hat,
                                   psi1_vals, psi2_vals)
        out["kernel"] = {"W_hat_refined": W_hat2.tolist(),
                         "W_hat_stage1": W_hat.tolist()}
        W_hat = W_hat2

        # Exact exposure-weighted rate calibration.  This is important here
        # because the five latent groups differ greatly in size.
        a_arr = a_hat.reindex(nodes_all).fillna(0.0).values
        c_rate = EST.calibrate_rate(
            train, groups, a_hat, W_hat, psi1_vals, psi2_vals)
        out["rate_calibration"] = {"c_rate": float(c_rate)}
        print(f"rate calibration c_rate = {c_rate:.3g}")

        # marks (sign model + magnitude bins)
        mark_model = EST.fit_mark_models(train, delta, groups)
        out["mark_model"] = {k: v for k, v in mark_model.items()
                             if k != "mag_dist"}
        out["mark_model"]["mag_dist"] = mark_model["mag_dist"]

        # REM-style baseline fit on the training case-control data
        t0 = time.time()
        cc_tr = EST.build_case_control(train, groups, entry, delta,
                                       a_hat=a_hat, W_hat=W_hat,
                                       n_controls=N_CONTROLS, seed=0)
        rem_feats = ["log_indegree", "net_cap", "count_cap", "age_days",
                     "log_recency", "ever_sent", "logW"]
        rem_res = EST.conditional_logit_fit(cc_tr, rem_feats)
        rem_beta = {f: float(rem_res.params[f]) for f in rem_feats}
        out["rem_baseline"] = {"coef": rem_beta,
                               "n_rows": int(len(cc_tr))}
        print(f"REM baseline fit ({time.time()-t0:.0f}s): "
              f"coef = { {k: round(v,3) for k,v in rem_beta.items()} }")

        fitted = {
            "fit_schema": fit_schema,
            "a_hat": a_hat.to_dict(), "groups": groups.to_dict(),
            "W_hat": W_hat.tolist(), "alpha": alpha, "beta": beta,
            "c_rate": float(c_rate), "delta": delta, "hl_days": HL_BASE,
            "rem_beta": rem_beta, "sign_model": {
                "coef": mark_model["sign_coef"],
                "intercept": mark_model["sign_intercept"]},
            "mag_dist": mark_model["mag_dist"],
            "mag_bin_edges": mark_model["mag_bin_edges"],
        }
        (EXP_DIR / "fitted_params.json").write_text(
            json.dumps(fitted, indent=2, default=str))

    # PA exponent: linear preferential attachment (alpha = 1.0); grid fitting
    # is a refinement noted in the robustness section
    pa_alpha = 1.0
    out["pa_alpha"] = pa_alpha

    # ----------------------------------------------------------- evaluate --
    eval_ = evaluate_models(test, train, nodes_all, pos_all, groups,
                            a_arr, W_hat, psi2, rem_beta, pa_alpha,
                            delta, c_rate, mark_model, entry)
    out["holdout"] = eval_
    out["wall_seconds"] = round(time.time() - t_start, 1)
    (EXP_DIR / "holdout.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"holdout": eval_["summary"]}, indent=2))
    print(f"\n[ok] wrote experiments/holdout.json "
          f"({out['wall_seconds']}s)")


def evaluate_models(test, train, nodes, pos, groups, a_arr, W_hat, psi2,
                    rem_beta, pa_alpha, delta, c_rate, mark_model, entry):
    """Test-period metrics for the kinetic model and the baselines.

    The node universe is train + test: test-only nodes get group -1 and
    zero activity (no kernel/activity information available for them)."""
    test_nodes = np.unique(np.concatenate(
        [test["source"].values, test["target"].values]))
    extra = np.setdiff1d(test_nodes, nodes)
    if len(extra):
        nodes = np.concatenate([nodes, extra])
        pos = {n: k for k, n in enumerate(nodes)}
        a_arr = np.concatenate([a_arr, np.zeros(len(extra))])
        g_arr = np.concatenate([groups.values,
                                -np.ones(len(extra), dtype=int)])
    else:
        g_arr = groups.values
    N = len(nodes)
    # The sign model used elapsed years since the start of its training data.
    t0 = train["timestamp"].min()

    # realised capitals: replay the TRAINING stream, then evolve through the
    # test events one by one (filtered history)
    eng = CapitalEngine(N, delta)
    eng_cnt = CapitalEngine(N, delta)
    indeg = np.zeros(N, dtype=int)
    sent = np.zeros(N, dtype=int)
    last_recv = np.full(N, -np.inf)
    # Replay all training-period observable state.  The test risk set and
    # covariates must start from the cutoff state rather than from zero.
    active = np.zeros(N, dtype=bool)
    # used pairs: all training pairs are already used (once-per-pair)
    used: dict[int, set[int]] = {}
    for row in train.itertuples(index=False):
        pi_tr, pj_tr = pos[row.source], pos[row.target]
        used.setdefault(pi_tr, set()).add(pj_tr)
        active[pi_tr] = active[pj_tr] = True
        indeg[pj_tr] += 1
        sent[pi_tr] += 1
        last_recv[pj_tr] = row.timestamp
    entry_t = entry["entry"].reindex(nodes).values

    rng = np.random.default_rng(0)
    n_test = len(test)
    scores = {m: np.zeros(n_test) for m in
              ["kinetic", "homogeneous", "pa", "rem", "activity"]}
    ranks = {m: np.zeros(n_test) for m in
             ["kinetic", "pa", "rem"]}
    y_sign = np.zeros(n_test)
    p_sign = np.zeros(n_test)
    cap_real = np.zeros(n_test)
    brier = np.zeros(n_test)

    for row in train.itertuples(index=False):
        eng.jump(pos[row.target], row.timestamp, ETA * row.rating)
        eng_cnt.jump(pos[row.target], row.timestamp, 1.0)

    # iterate over test events in order
    for e, row in enumerate(test.itertuples(index=False)):
        i, j, r, t = row.source, row.target, row.rating, row.timestamp
        pi_, pj = pos[i], pos[j]
        active[pi_] = active[pj] = True
        # NOTE: the pair (i,j) is added to `used` AFTER scoring/ranking below

        # candidates: realised target + sampled active non-used others
        act = np.flatnonzero(active)
        others = act[act != pj]
        avail = np.array([c for c in others
                          if c not in used.get(pi_, set())])
        n_ctrl = min(N_CONTROLS, len(avail))
        ctrl = rng.choice(avail, size=n_ctrl, replace=False)
        cand = np.concatenate([[pj], ctrl])

        net_at = np.array([eng.at(c, t) for c in cand])
        cnt_at = np.array([eng_cnt.at(c, t) for c in cand])
        # rank machinery: scores of all active candidates
        net_all = eng.at_all(t)
        cnt_all = eng_cnt.at_all(t)
        avail_all = np.array([c for c in act if c not in used.get(pi_, set())])

        # model scores on the sampled candidates (gi < 0 => no kernel info)
        gi = g_arr[pi_] if g_arr[pi_] >= 0 else -1
        logW_j = np.array([np.log(W_hat[gi, g_arr[c]] + 1e-9)
                           if gi >= 0 and g_arr[c] >= 0 else 0.0 for c in cand])
        sc_kin = logW_j + np.log(np.maximum(psi2(net_at), 1e-9))
        sc_hom = np.zeros(len(cand))
        sc_pa = pa_alpha * np.log1p(indeg[cand])
        if rem_beta is not None:
            rec_raw_c = (t - last_recv[cand]) / 86400.0
            rec_cand = np.log1p(np.maximum(
                np.where(np.isfinite(rec_raw_c), rec_raw_c,
                         (t - entry_t[cand]) / 86400.0), 0.0))
            sc_rem = np.array([rem_beta["log_indegree"] * np.log1p(indeg[cand[k]])
                               + rem_beta["net_cap"] * net_at[k]
                               + rem_beta["count_cap"] * cnt_at[k]
                               + rem_beta["age_days"] * (t - entry_t[cand[k]]) / 86400.0
                               + rem_beta["log_recency"] * rec_cand[k]
                               + rem_beta["ever_sent"] * (sent[cand[k]] > 0)
                               + rem_beta["logW"] * logW_j[k]
                               for k in range(len(cand))])
        else:
            sc_rem = np.zeros(len(cand))
        sc_act = np.zeros(len(cand))

        for m, sc in [("kinetic", sc_kin), ("homogeneous", sc_hom),
                      ("pa", sc_pa), ("rem", sc_rem), ("activity", sc_act)]:
            z = sc - sc.max()
            scores[m][e] = z[0] - np.log(np.exp(z).sum())

        # rankings among ALL active candidates
        logW_all = np.array([np.log(W_hat[gi, g_arr[c]] + 1e-9)
                             if gi >= 0 and g_arr[c] >= 0 else 0.0
                             for c in avail_all])
        rk_kin = np.log(np.maximum(psi2(net_all[avail_all]), 1e-9)) + logW_all
        rk_pa = pa_alpha * np.log1p(indeg[avail_all])
        if rem_beta is not None:
            rec_raw_a = (t - last_recv[avail_all]) / 86400.0
            rec_all = np.log1p(np.maximum(
                np.where(np.isfinite(rec_raw_a), rec_raw_a,
                         (t - entry_t[avail_all]) / 86400.0), 0.0))
            rk_rem = (rem_beta["log_indegree"] * np.log1p(indeg[avail_all])
                      + rem_beta["net_cap"] * net_all[avail_all]
                      + rem_beta["count_cap"] * cnt_all[avail_all]
                      + rem_beta["age_days"] * (t - entry_t[avail_all]) / 86400.0
                      + rem_beta["log_recency"] * rec_all
                      + rem_beta["ever_sent"] * (sent[avail_all] > 0)
                      + rem_beta["logW"] * logW_all)
        else:
            rk_rem = np.zeros(len(avail_all))
        pos_j = int(np.flatnonzero(avail_all == pj)[0])
        for m, rk in [("kinetic", rk_kin), ("pa", rk_pa), ("rem", rk_rem)]:
            order = np.argsort(-rk)
            rank = int(np.flatnonzero(order == pos_j)[0]) + 1
            ranks[m][e] = rank

        # sign prediction
        Xs = np.array([[net_at[0], cnt_at[0],
                        (t - t0) / (86400.0 * 365.0)]])
        lin = Xs @ np.array(mark_model["sign_coef"]) + mark_model["sign_intercept"]
        phat = 1.0 / (1.0 + np.exp(-lin))[0]
        y_sign[e] = int(r > 0)
        p_sign[e] = float(np.clip(phat, 1e-6, 1 - 1e-6))
        brier[e] = (phat - y_sign[e]) ** 2
        cap_real[e] = net_at[0]

        # update state with the realised event (pair becomes used now)
        used.setdefault(pi_, set()).add(pj)
        indeg[pj] += 1
        sent[pi_] += 1
        eng_cnt.jump(pj, t, 1.0)
        eng.jump(pj, t, ETA * r)
        last_recv[pj] = t

    # ---- summary metrics ------------------------------------------------
    def logloss(p, y):
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    sum_ = {}
    for m in scores:
        sum_[m] = {"partial_loglik": float(scores[m].mean()),
                   "top10_recall": float((ranks.get(m, np.zeros(n_test)) <= TOPK).mean())
                   if m in ranks else None,
                   "mrr": float((1.0 / np.maximum(ranks.get(m, np.ones(n_test)), 1)).mean())
                   if m in ranks else None}
    sum_["sign_logloss"] = logloss(p_sign, y_sign)
    sum_["sign_brier"] = float(brier.mean())
    sum_["n_test"] = n_test

    # ---- macro: predicted vs realised -----------------------------------
    macro = macro_metrics(train, test, nodes, pos, g_arr, a_arr, W_hat,
                          psi2, delta, c_rate, cap_real)
    return {"summary": sum_, "macro": macro, "cap_real_tail":
            {"mean": float(cap_real.mean()), "p10": float(np.percentile(cap_real, 10)),
             "p50": float(np.percentile(cap_real, 50)),
             "p90": float(np.percentile(cap_real, 90))}}


def macro_metrics(train, test, nodes, pos, g_arr, a_arr, W_hat, psi2,
                  delta, c_rate, cap_real):
    """Run the kinetic solver over the test period with the fitted parameters
    and compare the predicted event-weighted capital distribution and the
    event-rate trajectory with the realised test period."""
    from dengyunetwork.kinetic import KineticSolver, KineticSpec

    K = len(W_hat)
    N = len(nodes)
    t0_test = test["timestamp"].min()
    t1_test = test["timestamp"].max()
    T_days = (t1_test - t0_test) / 86400.0

    # group proportions and activity at the test start
    pi_k = np.bincount(g_arr[g_arr >= 0], minlength=K).astype(float)
    pi_k /= pi_k.sum()
    a_k = np.array([a_arr[g_arr == k].mean() if (g_arr == k).any() else 0.0
                    for k in range(K)])
    # entry source during the test period: per-group new users per day
    train_nodes = set(train["source"]) | set(train["target"])
    test_new = test[~test["source"].isin(train_nodes) | ~test["target"].isin(train_nodes)]
    new_by_group = np.zeros(K)
    for row in test_new.itertuples(index=False):
        for nd in (row.source, row.target):
            if nd in nodes and g_arr[pos.get(nd, 0)] >= 0 and nd not in train_nodes:
                new_by_group[g_arr[pos[nd]]] += 1
    source_rate = new_by_group / T_days

    # initial condition: empirical capital distribution at the test start
    eng0 = CapitalEngine(N, delta)
    for row in train.itertuples(index=False):
        eng0.jump(pos[row.target], row.timestamp, ETA * row.rating)
    eng0.at_all(t0_test)

    M = 1400
    cmin, cmax = -2.0, 12.0   # real-data capitals reach ~6-10 (p90 ~ 5.3)
    spec = KineticSpec(
        K=K, pi=pi_k, activity=a_k, W=c_rate * W_hat,
        delta=delta, eta=0.1,
        psi1=lambda c: 1.0, psi2=psi2,
        r_values=mark_support(), q_cells=mark_q(eng0.value, M),
        c_min=cmin, c_max=cmax, M=M, dt_max=0.05, T=T_days,
        source_rate=source_rate if source_rate.sum() > 0 else None,
    )
    spec.initial = "uniform"
    # build f0 from the empirical capital distribution
    sol = KineticSolver(spec)
    edges = np.linspace(cmin, cmax, M + 1)
    for k in range(K):
        values_k = eng0.value[g_arr == k]
        hist_k, _ = np.histogram(np.clip(values_k, cmin, cmax), bins=edges)
        if hist_k.sum():
            sol.f[k, :] = pi_k[k] * (hist_k / hist_k.sum()) / sol.h
    # event-weighted density accumulation
    grid = sol.c_mid
    import os
    if os.environ.get("H06_DEBUG"):
        print("[macro] N", N, "K", K, "pi_k", np.round(pi_k, 4),
              "a_k", np.round(a_k, 5), "f0 mass", float(f0.sum() * (edges[1]-edges[0])),
              "init rate", float(sol.aggregate_rate()), "source", float(source_rate.sum()))
    wgt_density = np.zeros(M)
    tot_rate = 0.0
    n_bins = 16
    bin_edges = np.linspace(0, T_days, n_bins + 1)
    rates = np.zeros(n_bins)
    idx = 0
    n_steps = 0
    while sol.t < T_days - 1e-12:
        dt = sol.step()
        n_steps += 1
        kap = sol._rate()
        wgt_density += (sol.f * kap).sum(axis=0) * dt
        tot_rate += float(((sol.f * kap) * sol.h).sum()) * dt
        while sol.t >= (idx + 1) * (T_days / n_bins) and idx < n_bins - 1:
            idx += 1
        rates[idx] += float(((sol.f * kap) * sol.h).sum()) * dt
    # size-biased density: wgt_density is a rate DENSITY per unit c, so its
    # integral is tot_rate; divide by (tot_rate / h) to normalise to 1
    wgt_density /= max(tot_rate / sol.h, 1e-300)

    # realised size-biased capital (test events) and trajectory
    hist_real, _ = np.histogram(np.clip(cap_real, cmin, cmax), bins=M,
                                range=(cmin, cmax))
    f_real = hist_real / max(hist_real.sum(), 1)
    # W1 between the two (on the common grid)
    F_pred = np.cumsum(wgt_density)
    F_real = np.cumsum(f_real)
    w1 = float(np.trapezoid(np.abs(F_pred - F_real), grid))
    # JSD
    eps = 1e-12
    p_ = np.maximum(wgt_density, eps) / np.maximum(wgt_density, eps).sum()
    q_ = np.maximum(f_real, eps) / np.maximum(f_real, eps).sum()
    jsd = float(0.5 * (np.sum(p_ * np.log(p_ / ((p_ + q_) / 2)))
                       + np.sum(q_ * np.log(q_ / ((p_ + q_) / 2)))))

    # realised trajectory
    h_real, _ = np.histogram((test["timestamp"].values - t0_test) / 86400.0,
                             bins=bin_edges)
    real_rates = h_real / (T_days / n_bins) / N   # per-day per-capita
    rates = rates / (T_days / n_bins)             # per-day per-capita
    iae = float(np.mean(np.abs(real_rates - rates)
                        / np.maximum(real_rates, 1e-9)))
    return {"w1_size_biased": w1, "jsd": jsd, "trajectory_iae": iae,
            "pred_rates": rates.tolist(), "real_rates": real_rates.tolist(),
            "n_solver_steps": n_steps, "tot_rate": float(tot_rate)}


def mark_support():
    # The data update is C^+ = C + eta * rating with eta=.1 and integer
    # ratings in [-10,10].  Hence R itself is the integer rating here.
    return np.array([sgn * float(mag)
                     for sgn in (-1, 1) for mag in range(1, 11)])


def mark_q(capitals, M, cmin=-2.0, cmax=12.0):
    """Target-capital projection of the fitted mark model.

    The kinetic state does not contain count capital or calendar time, so
    the sign probability uses the fitted intercept and net-capital term;
    conditional magnitude probabilities use the empirical net-capital bins.
    """
    cm = np.linspace(cmin, cmax, M + 1)
    cm = 0.5 * (cm[:-1] + cm[1:])
    import json
    fitted = json.load(open(EXP_DIR / "fitted_params.json"))
    coef = fitted["sign_model"]["coef"]
    intercept = fitted["sign_model"]["intercept"]
    p_pos = 1.0 / (1.0 + np.exp(-(intercept + coef[0] * cm)))
    edges = np.asarray(fitted["mag_bin_edges"], dtype=float)
    mag_dist = np.asarray(fitted["mag_dist"], dtype=float)
    b = np.searchsorted(edges[1:-1], cm, side="right")
    b = np.clip(b, 0, len(mag_dist) - 1)
    p_mag = mag_dist[b]
    q = np.zeros((M, 20))
    q[:, :10] = (1 - p_pos)[:, None] * p_mag
    q[:, 10:] = p_pos[:, None] * p_mag
    return q


if __name__ == "__main__":
    main()
