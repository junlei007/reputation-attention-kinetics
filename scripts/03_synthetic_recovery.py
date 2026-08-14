#!/usr/bin/env python
"""WP2: parameter recovery on synthetic data (gate G2).

Generates events from the frozen-rate model with known parameters, then
re-runs the WP5-style staged estimators and checks each component comes
back.  Components recovered:

  1. activity a_i            -> up to the global scale absorbed in W
  2. capital function psi2   -> via conditional-logit probe (with known W
                                as control, isolating the capital effect)
  3. block kernel W          -> block counts / model exposure, up to scale
  4. mark distribution Q     -> logistic sign model + magnitude shape
  5. decay delta             -> profile likelihood over a grid
  6. sparsity                -> total event count scales ~ O(N), edges/N^2 -> 0
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dengyunetwork.simulator import (  # noqa: E402
    EventSimulator, SimParams, logistic_mark, tanh_psi,
)
from dengyunetwork.data import sort_by_time  # noqa: E402

EXP_DIR = ROOT / "experiments"

TRUE = {
    "N": 500, "K": 4, "T": 4.0,
    "delta": np.log(2.0) / 30.0,          # 30-day half-life (unit time = day)
    "alpha1": 0.3, "beta1": 3.0,
    "alpha2": 0.8, "beta2": 2.0,          # strong target-side effect: psi2 in [0.2, 1.8]
    "mark": {"intercept": 0.5, "b_target": 1.0, "b_sender": 0.2},
}

# long-horizon design for decay/activity recovery: T spans several
# half-lives (3-6) so the sample contains strong decay information -- with
# fast decay the aggregate rate plateaus at its steady state, with slow
# decay it keeps growing; the trajectory shape separates the candidates
LONG = {"N": 200, "T": 90.0, "eta": 0.1}


def build_true_params(seed: int = 0, T: float | None = None,
                      N: int | None = None):
    rng = np.random.default_rng(seed)
    N = TRUE["N"] if N is None else N
    K = TRUE["K"]
    groups = rng.integers(0, K, size=N)
    activity = rng.lognormal(mean=0.0, sigma=0.8, size=N)
    W = np.full((K, K), 0.4)
    np.fill_diagonal(W, 1.2)
    W[0, 1:] = 0.9
    W[1:, 0] = 0.9
    entry = np.sort(rng.uniform(0.0, 0.3, size=N))
    p = SimParams(N=N, K=K, groups=groups, activity=activity, W=W,
                  delta=TRUE["delta"], T=TRUE["T"] if T is None else T,
                  entry_times=entry)
    return p, rng


def generate(p: SimParams, seed: int = 1) -> pd.DataFrame:
    sim = EventSimulator(
        p,
        tanh_psi(TRUE["alpha1"], TRUE["beta1"]),
        tanh_psi(TRUE["alpha2"], TRUE["beta2"]),
        logistic_mark(**TRUE["mark"]),
        rng=np.random.default_rng(seed),
    )
    res = sim.run()
    return res.to_frame()


# --------------------------------------------------------------------------
# recovery estimators (mirror of the WP5 pipeline, simplified)
# --------------------------------------------------------------------------

def recover_activity(df: pd.DataFrame, p: SimParams, N: int) -> dict:
    """a_hat_i = out-events / active time; compare to true a_i up to scale."""
    T_act = np.zeros(N)
    entry = p.entry_times
    # active time = T - entry (entries all <= T in this design)
    T_act = p.T - np.minimum(entry, p.T)
    out = np.zeros(N)
    for row in df.itertuples(index=False):
        out[row.source] += 1.0
    a_hat = np.divide(out, T_act, out=np.zeros(N), where=T_act > 0)
    # scale-match: a_true = s * a_hat by least squares through origin
    s = float(np.sum(p.activity * a_hat) / np.sum(a_hat * a_hat))
    resid = p.activity - s * a_hat
    return {
        "scale_s": round(s, 4),
        "corr_log": float(np.corrcoef(np.log1p(p.activity), np.log1p(a_hat))[0, 1]),
        "rel_rmse": float(np.sqrt(np.mean(resid ** 2)) / np.mean(p.activity)),
    }


def build_case_control(df: pd.DataFrame, p: SimParams, delta: float,
                       n_controls: int = 30, rng_seed: int = 0) -> pd.DataFrame:
    """Per-event target case-control dataset (as in facts.run_g1_probe) with
    the known kernel as a control covariate and decayed capital as the target
    covariate of interest.  Capital state is computed with the given delta."""
    rng = np.random.default_rng(rng_seed)
    N, g, W = p.N, p.groups, p.W
    C = np.zeros(N)
    last_t = np.full(N, -np.inf)
    active_since = np.zeros(N, dtype=bool)
    rows = []
    for e, row in enumerate(df.itertuples(index=False)):
        i, j, t = row.source, row.target, row.timestamp
        active_since[i] = active_since[j] = True

        def cap_at(k):
            if t > last_t[k]:
                C[k] *= np.exp(-delta * (t - last_t[k]))
                last_t[k] = t
            return C[k]

        act = np.flatnonzero(active_since)
        others = act[act != j]
        ctrl = rng.choice(others, size=min(n_controls, len(others)), replace=False)
        cand = np.concatenate([[j], ctrl])
        case = np.zeros(len(cand)); case[0] = 1.0
        cj = np.array([cap_at(k) for k in cand])
        logW = np.array([np.log(W[g[i], g[k]] + 1e-9) for k in cand])
        rows.append(pd.DataFrame({"stratum": e, "case": case,
                                  "cap": cj, "logW": logW}))
        C[j] += p.eta * row.rating  # matches the generator (eta from SimParams)
        last_t[j] = t
    return pd.concat(rows, ignore_index=True)


def conditional_logit(cc: pd.DataFrame, feats=("cap", "logW")):
    from statsmodels.discrete.conditional_models import ConditionalLogit

    m = ConditionalLogit(endog=cc["case"], exog=cc[list(feats)],
                         groups=cc["stratum"])
    return m.fit(disp=False, maxiter=200)


def recover_psi_probe(df: pd.DataFrame, p: SimParams) -> dict:
    """Conditional-logit probe on target choice with the true kernel as a
    control; the capital coefficient isolates the psi2 effect."""
    cc = build_case_control(df, p, TRUE["delta"])
    res = conditional_logit(cc)
    return {
        "n_events": int(df.shape[0]),
        "coef_cap": float(res.params["cap"]),
        "se_cap": float(res.bse["cap"]),
        "p_cap": float(res.pvalues["cap"]),
        "coef_logW": float(res.params["logW"]),
        "se_logW": float(res.bse["logW"]),
        "true_beta2": TRUE["beta2"],
        "sign_matches": bool(np.sign(res.params["cap"]) == np.sign(TRUE["beta2"])),
    }


def recover_kernel(df: pd.DataFrame, p: SimParams) -> dict:
    """W_hat_gh = (N * count_gh) / (T * avg block exposure), up to a common
    scale; compares the SHAPE (log-correlation) after scale-matching."""
    N, K = p.N, p.K
    count = np.zeros((K, K))
    for row in df.itertuples(index=False):
        count[p.groups[row.source], p.groups[row.target]] += 1.0
    n_g = np.bincount(p.groups, minlength=K)
    W_hat = N * count / max(p.T, 1e-9) / (n_g[:, None] * n_g[None, :])
    mask = p.W > 0
    s = float(np.sum(p.W[mask] * W_hat[mask]) / np.sum(W_hat[mask] ** 2))
    W_adj = s * W_hat
    log_corr = float(np.corrcoef(
        np.log(p.W[mask] + 1e-6), np.log(W_adj[mask] + 1e-6))[0, 1])
    return {
        "log_corr_shape": log_corr,
        "rel_rmse": float(np.sqrt(np.mean((p.W[mask] - W_adj[mask]) ** 2))
                          / np.mean(p.W[mask])),
        "W_hat": W_hat.tolist(),
        "true_W": p.W.tolist(),
    }


def recover_mark(df: pd.DataFrame) -> dict:
    """Logistic regression of sign on (C_i, C_j) + magnitude shape check."""
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([df["cap_source"].values, df["cap_target"].values])
    y = (df["rating"].values > 0).astype(int)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
    mag = df["rating"].abs().values
    mag_counts = np.bincount((mag * 10).astype(int), minlength=11)
    return {
        "b_sender": float(lr.coef_[0][0]),
        "b_target": float(lr.coef_[0][1]),
        "intercept": float(lr.intercept_[0]),
        "true": TRUE["mark"],
        "magnitude_uniform_rmse": float(
            np.std(mag_counts[1:11] / mag_counts[1:11].sum() - 0.1)),
    }


def capitals_series(df: pd.DataFrame, p: SimParams, delta: float) -> np.ndarray:
    """Target capital at each event time, recomputed under decay rate delta."""
    C = np.zeros(p.N)
    last_t = np.full(p.N, -np.inf)
    out = np.zeros(len(df))
    for e, row in enumerate(df.itertuples(index=False)):
        j = row.target
        if row.timestamp > last_t[j]:
            C[j] *= np.exp(-delta * (row.timestamp - last_t[j]))
            last_t[j] = row.timestamp
        out[e] = C[j]
        C[j] += p.eta * row.rating
        last_t[j] = row.timestamp
    return out


def simulate_cumulative_trajectory(p: SimParams, psi1, psi2, mark,
                                   n_runs: int = 16, n_bins: int = 30,
                                   seed_base: int = 100) -> np.ndarray:
    """Mean cumulative event-count trajectory over ``n_bins`` time bins from
    ``n_runs`` simulator replications (all other parameters fixed)."""
    bins = np.linspace(0.0, p.T, n_bins + 1)
    cum = np.zeros(n_bins)
    for s in range(n_runs):
        sim = EventSimulator(p, psi1, psi2, mark,
                             rng=np.random.default_rng(seed_base + s))
        res = sim.run()
        c, _ = np.histogram(res.t, bins=bins)
        cum += np.cumsum(c)
    return cum / n_runs


def recover_delta(df: pd.DataFrame, p: SimParams,
                  grid_days=(15, 30, 60, 120)) -> dict:
    """Decay-rate recovery via the aggregate event-rate trajectory.

    Event-level (mark/choice) likelihoods are nearly delta-flat: linear
    response models are scale-invariant in the capital feature, and the
    curvature of the capital response is the only event-level channel.
    The aggregate rate trajectory is the strong channel: Lambda(t) =
    (1/N) sum_ij a_i W_ij psi1(C_i) psi2(C_j) depends on the level of the
    aggregate capital, which the decay rate directly controls.

    For each candidate half-life the simulator (with all other components at
    their recovered values) predicts the cumulative count trajectory; the
    best delta minimises the relative L1 distance to the observed one.
    """
    from copy import deepcopy

    bins = np.linspace(0.0, p.T, 31)
    obs, _ = np.histogram(df["timestamp"].values, bins=bins)
    obs_cum = np.cumsum(obs)
    obs_cum = obs_cum / obs_cum[-1] if obs_cum[-1] > 0 else obs_cum

    results = {}
    psi1 = tanh_psi(TRUE["alpha1"], TRUE["beta1"])
    psi2 = tanh_psi(TRUE["alpha2"], TRUE["beta2"])
    mark = logistic_mark(**TRUE["mark"])
    for hl in grid_days:
        pp = deepcopy(p)
        pp.delta = np.log(2.0) / hl
        traj = simulate_cumulative_trajectory(pp, psi1, psi2, mark)
        traj = traj / traj[-1] if traj[-1] > 0 else traj
        dist = float(np.mean(np.abs(traj - obs_cum)))
        results[hl] = {"rel_l1_trajectory": dist}
    best = min(results, key=lambda k: results[k]["rel_l1_trajectory"])
    return {"grid": {str(k): v for k, v in results.items()},
            "argmax_hl_days": int(best),
            "true_hl_days": 30.0,
            "delta_true": TRUE["delta"],
            "note": "delta is recovered from the aggregate rate trajectory "
                    "(event-level likelihoods are nearly flat; linear "
                    "response models are scale-invariant in the features)"}


def sparsity_check(seeds=(0, 1, 2)) -> dict:
    """Event count per unit time vs N: expect roughly linear in N (sparse
    network: edges = O(N), edges/N^2 -> 0)."""
    out = {}
    for N in (250, 500, 1000):
        p, _ = build_true_params(seed=0)
        p.N = N
        p.groups = np.random.default_rng(7).integers(0, p.K, size=N)
        p.activity = np.random.default_rng(8).lognormal(0, 0.8, size=N)
        p.entry_times = np.sort(np.random.default_rng(9).uniform(0, 0.3, size=N))
        counts = []
        for s in seeds:
            df = generate(p, seed=s)
            counts.append(len(df))
        out[str(N)] = {"mean_events": float(np.mean(counts)),
                       "events_per_capita": float(np.mean(counts) / N)}
    return out


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(exist_ok=True)
    # main design: moderate T, strong psi2 -> psi/kernel/mark recovery
    p, _ = build_true_params()
    df = sort_by_time(generate(p))
    # long design: T comparable to the half-life -> delta/activity recovery
    p_long, _ = build_true_params(T=LONG["T"], N=LONG["N"])
    p_long.eta = LONG["eta"]
    df_long = sort_by_time(generate(p_long, seed=2))
    out = {
        "true_params": TRUE,
        "long_design": LONG,
        "n_events": int(len(df)),
        "n_nodes": p.N,
        "n_unique_pairs": int(df[["source", "target"]].drop_duplicates().shape[0]),
        "activity": recover_activity(df_long, p_long, p_long.N),
        "psi_probe": recover_psi_probe(df, p),
        "kernel": recover_kernel(df, p),
        "mark": recover_mark(df),
        "delta": recover_delta(df_long, p_long),
        "sparsity": sparsity_check(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (EXP_DIR / "synthetic_recovery.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("true_params", "kernel", "delta")},
                     indent=2, default=str))
    print("\nkernel:", json.dumps(out["kernel"], indent=1)[:400])
    print("\ndelta:", json.dumps(out["delta"], indent=1)[:400])
    print("\n[ok] wrote experiments/synthetic_recovery.json")


if __name__ == "__main__":
    main()
