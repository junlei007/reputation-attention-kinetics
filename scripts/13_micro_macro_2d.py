#!/usr/bin/env python
"""WP3 (2D, v0.11): micro-macro consistency for the joint (C, H) model.

The frozen v0.11 theorem concerns the two-dimensional state X = (C, H).
This script validates the two-dimensional numerical pair:

  * the particle simulator with dH = -rho H dt + dN_in, target response
    psi_C(C_j) psi_H(H_j)  (src/dengyunetwork/simulator.py);
  * the deterministic finite-volume solver for the joint K x C x H law
    (src/dengyunetwork/kinetic.py, KineticSolver2D).

Checks (mirroring the one-dimensional gate G3 design of 04_micro_macro.py):
  1. W1 of the empirical CAPITAL and ATTENTION marginals of the N-particle
     process to the solver marginals decays ~ N^{-1/2} as N grows
     (Theorem 3 marginal rates);
  2. sliced-W1 diagnostic of the JOINT empirical measure across N (the
     critical-dimension bound of Theorem 3 carries log(1+N)/sqrt(N));
  3. per-group mass conservation of the solver (raw drift before the
     renormalisation step is tracked and reported);
  4. per-capita event-rate trajectory of MC (binned) matches the solver's
     predicted trajectory;
  5. solver wall-clock vs one MC run at the largest N.

All numbers written to experiments/micro_macro_2d.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dengyunetwork.kinetic import KineticSpec2D, KineticSolver2D  # noqa: E402
from dengyunetwork.simulator import (  # noqa: E402
    EventSimulator, SimParams, logistic_mark, tanh_psi,
)

EXP_DIR = ROOT / "experiments"

# Block design identical to 04_micro_macro.py, augmented with the attention
# dynamics: rho = 0.5 (H half-life ~ 1.39 time units) and the bounded
# attention response psi_H(h) = 1 + 0.6 tanh(h).
PARAMS = {
    "K": 4, "T": 4.0,
    "delta": np.log(2.0) / 30.0,
    "rho": 0.5,
    "alpha1": 0.3, "beta1": 3.0,     # psi1 (C-only instance)
    "alphaC": 0.8, "betaC": 2.0,     # psi_C
    "alphaH": 0.6, "betaH": 1.0,     # psi_H
    "eta": 0.1,
    "activity": np.array([0.4, 0.8, 1.5, 2.5]),   # deterministic per group
    "mark": {"intercept": 0.5, "b_target": 1.0, "b_sender": 0.0},
}
W_TRUE = np.full((PARAMS["K"], PARAMS["K"]), 0.4)
np.fill_diagonal(W_TRUE, 1.2)
W_TRUE[0, 1:] = 0.9
W_TRUE[1:, 0] = 0.9

M_C, H_MAX, M_H = 1000, 16.0, 400  # dh = 0.04: the +1 attention jump is exact
# (resolution diagnostic: Mh160/dt0.02 gives W1 biases ~0.018 (C) and ~0.26
#  (H); Mh400/dt0.005 gives ~0.009 and ~0.10, at which the residual is the
#  decay-then-jump time-splitting error, absorbed as the fit floor)


def build_mc_params(N: int, groups: np.ndarray) -> SimParams:
    return SimParams(
        N=N, K=PARAMS["K"], groups=groups,
        activity=PARAMS["activity"][groups].copy(),
        W=W_TRUE, delta=PARAMS["delta"], rho=PARAMS["rho"],
        eta=PARAMS["eta"], T=PARAMS["T"], entry_times=np.zeros(N))


def responses():
    return (tanh_psi(PARAMS["alpha1"], PARAMS["beta1"]),
            tanh_psi(PARAMS["alphaC"], PARAMS["betaC"]),
            tanh_psi(PARAMS["alphaH"], PARAMS["betaH"]))


def build_spec(pi: np.ndarray, dt_max: float = 0.005) -> KineticSpec2D:
    K = PARAMS["K"]
    c_mid = 0.5 * (np.linspace(-2.0, 2.0, M_C + 1)[:-1]
                   + np.linspace(-2.0, 2.0, M_C + 1)[1:])
    r_vals = np.array([sgn * mag / 10.0
                       for sgn in (-1, 1) for mag in range(1, 11)])
    p_pos = 1.0 / (1.0 + np.exp(-(PARAMS["mark"]["intercept"]
                                  + PARAMS["mark"]["b_target"] * c_mid)))
    q = np.zeros((M_C, 20))
    q[:, :10] = ((1.0 - p_pos) / 10.0)[:, None]
    q[:, 10:] = (p_pos / 10.0)[:, None]
    psi1, psiC, psiH = responses()
    return KineticSpec2D(
        K=K, pi=pi, activity=PARAMS["activity"], W=W_TRUE,
        delta=PARAMS["delta"], rho=PARAMS["rho"], eta=PARAMS["eta"],
        psi1=psi1, psiC=psiC, psiH=psiH,
        r_values=r_vals, q_cells=q, c_min=-2.0, c_max=2.0,
        M_c=M_C, H_max=H_MAX, M_h=M_H, dt_max=dt_max, T=PARAMS["T"])


def run_mc(p: SimParams, n_runs: int, seed_base: int = 100
           ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns (per-run sorted capital marginals, per-run attention
    marginals) at time T, plus the full states for the sliced-W1 check."""
    psi1, psiC, psiH = responses()
    mark = logistic_mark(**PARAMS["mark"])
    states = []
    for s in range(n_runs):
        sim = EventSimulator(p, psi1, psiC, mark, psiH=psiH,
                             rng=np.random.default_rng(seed_base + s))
        sim.run()
        states.append((sim.C.copy(), sim.H.copy()))
    return states


def w1_cdf(sorted_x: np.ndarray, f: np.ndarray, grid: np.ndarray) -> float:
    """W1 between an empirical sample and a density f on ``grid`` (midpoints)."""
    F_emp = np.searchsorted(sorted_x, grid) / len(sorted_x)
    F_mac = np.cumsum(f) / max(f.sum(), 1e-300)
    return float(np.trapezoid(np.abs(F_emp - F_mac), grid))


def w1_weighted(sorted_a: np.ndarray, wa: np.ndarray,
                sorted_b: np.ndarray, wb: np.ndarray) -> float:
    """W1 between two weighted empirical measures (sorted supports)."""
    if len(sorted_a) == 0 or len(sorted_b) == 0:
        return 0.0
    Fa = np.concatenate([[0.0], np.cumsum(wa)])
    Fb = np.concatenate([[0.0], np.cumsum(wb)])
    x = np.sort(np.concatenate([sorted_a, sorted_b]))
    ga = np.searchsorted(sorted_a, x, side="right")
    gb = np.searchsorted(sorted_b, x, side="right")
    Fa_x = Fa[ga]; Fb_x = Fb[gb]
    dx = np.diff(x)
    return float(np.abs(Fa_x[:-1] - Fb_x[:-1]) @ dx)


def sliced_w1(states_2d: np.ndarray, grid_c: np.ndarray, grid_h: np.ndarray,
              mass2d: np.ndarray, n_dir: int = 32) -> float:
    """Mean 1D W1 over n_dir projection directions of the joint (C, H)
    law (a standard sliced-W1 diagnostic; lower bound on the joint W1)."""
    N = len(states_2d)
    theta = np.linspace(0.0, np.pi, n_dir, endpoint=False)
    dirs = np.column_stack([np.cos(theta), np.sin(theta)])   # (n_dir, 2)
    # solver support: all (c, h) cell midpoints with their joint mass
    C, H = np.meshgrid(grid_c, grid_h, indexing="ij")
    supp = np.column_stack([C.ravel(), H.ravel()])
    mass = mass2d.ravel() / max(mass2d.sum(), 1e-300)
    keep = mass > 0
    supp = supp[keep]; mass = mass[keep]
    wa = np.full(N, 1.0 / N)
    tot = 0.0
    for d in dirs:
        pa = states_2d @ d
        pb = supp @ d
        oa = np.argsort(pa); ob = np.argsort(pb)
        tot += w1_weighted(pa[oa], wa[oa], pb[ob], mass[ob])
    return tot / n_dir


def main() -> None:
    EXP_DIR.mkdir(exist_ok=True)
    out: dict = {}
    t_start = time.time()

    n_blocks = np.array([50, 120, 180, 150])
    pi = n_blocks / n_blocks.sum()

    # macro solution (N-independent)
    spec = build_spec(pi)
    solver = KineticSolver2D(spec)
    t0 = time.time()
    solver.run()
    macro_wall = time.time() - t0
    f_c = solver.marginal_c() / solver.dc      # capital density on c_mid
    f_h = solver.marginal_h() / solver.dh      # attention density on h_mid
    mass2d = solver.f.sum(axis=0) * solver.dc * solver.dh  # (Mc, Mh)
    out["macro_wall_s"] = round(macro_wall, 2)
    out["macro_cells"] = int(spec.K * spec.M_c * spec.M_h)
    out["mass_after_run"] = {
        "per_group": (solver.f * solver.dc * solver.dh
                      ).sum(axis=(1, 2)).round(6).tolist(),
        "pi": pi.round(6).tolist(),
        "max_raw_drift_before_renorm": round(solver.raw_drift, 6),
    }

    sizes = [250, 500, 1000, 2000, 4000]
    n_runs = 8
    w1_results = []
    for N in sizes:
        pi_N = np.round(pi * N).astype(int)
        pi_N[-1] = N - pi_N[:-1].sum()
        groups = np.repeat(np.arange(4), pi_N)          # exact blocks
        p = build_mc_params(N, groups)
        t0 = time.time()
        states = run_mc(p, n_runs)
        mc_wall = (time.time() - t0) / n_runs
        w1_c, w1_h, sw1 = [], [], []
        for C_state, H_state in states:
            w1_c.append(w1_cdf(np.sort(C_state), f_c, solver.c_mid))
            w1_h.append(w1_cdf(np.sort(H_state), f_h, solver.h_mid))
            # per-run sliced-W1 of the N-particle joint empirical measure
            # (the theory object is one N-particle system, NOT the pooled
            # 8N sample: pooling would change the fluctuation scale)
            sw1.append(sliced_w1(np.column_stack([C_state, H_state]),
                                 solver.c_mid, solver.h_mid, mass2d))
        w1_results.append({
            "N": N,
            "w1_c_single_run_mean": float(np.mean(w1_c)),
            "w1_c_single_run_std": float(np.std(w1_c)),
            "w1_h_single_run_mean": float(np.mean(w1_h)),
            "w1_h_single_run_std": float(np.std(w1_h)),
            "sliced_w1_joint_mean": float(np.mean(sw1)),
            "sliced_w1_joint_std": float(np.std(sw1)),
            "mc_wall_per_run_s": round(mc_wall, 2),
        })
        print(f"N={N}: W1_C={np.mean(w1_c):.4f}(+-{np.std(w1_c):.4f}) "
              f"W1_H={np.mean(w1_h):.4f}(+-{np.std(w1_h):.4f}) "
              f"sW1_joint={np.mean(sw1):.4f}(+-{np.std(sw1):.4f}) "
              f"MC {mc_wall:.2f}s")

    # fits: W1(N) = a + b * N^{-1/2} for both marginals
    ns = np.array([r["N"] for r in w1_results])
    wc = np.array([r["w1_c_single_run_mean"] for r in w1_results])
    wh = np.array([r["w1_h_single_run_mean"] for r in w1_results])
    sw = np.array([r["sliced_w1_joint_mean"] for r in w1_results])
    sw_sd = np.array([r["sliced_w1_joint_std"] for r in w1_results])
    X_half = np.column_stack([np.ones_like(ns), ns ** -0.5])
    a_c, b_c = np.linalg.lstsq(X_half, wc, rcond=None)[0]
    a_h, b_h = np.linalg.lstsq(X_half, wh, rcond=None)[0]
    # joint: log(1+N)/sqrt(N) shape (critical-dimension rate)
    X_log = np.column_stack([np.ones_like(ns),
                             np.log(1.0 + ns) * ns ** -0.5])
    # residual comparison: which shape fits the joint diagnostic better?
    res_half = np.sqrt(np.mean((sw - X_half @ np.linalg.lstsq(
        X_half, sw, rcond=None)[0]) ** 2))
    res_log = np.sqrt(np.mean((sw - X_log @ np.linalg.lstsq(
        X_log, sw, rcond=None)[0]) ** 2))
    # marginal-fit slope standard errors: ordinary OLS on the five points
    # (n = 5), residual variance with n - 2 degrees of freedom
    def _ols_se(y: np.ndarray) -> float:
        beta = np.linalg.lstsq(X_half, y, rcond=None)[0]
        resid = y - X_half @ beta
        sigma2 = float(resid @ resid / (len(y) - X_half.shape[1]))
        cov = sigma2 * np.linalg.inv(X_half.T @ X_half)
        return float(np.sqrt(cov[1, 1]))

    se_c = _ols_se(wc)
    se_h = _ols_se(wh)
    out["w1_vs_n"] = w1_results
    out["fit_c_marginal"] = {"floor_a": round(float(a_c), 4),
                             "fluct_b": round(float(b_c), 4),
                             "fluct_b_se": round(se_c, 4),
                             "r2": round(float(1 - np.sum((wc - X_half @
                                     np.linalg.lstsq(X_half, wc,
                                                     rcond=None)[0]) ** 2)
                                     / np.sum((wc - wc.mean()) ** 2)), 3)}
    out["fit_h_marginal"] = {"floor_a": round(float(a_h), 4),
                             "fluct_b": round(float(b_h), 4),
                             "fluct_b_se": round(se_h, 4),
                             "r2": round(float(1 - np.sum((wh - X_half @
                                     np.linalg.lstsq(X_half, wh,
                                                     rcond=None)[0]) ** 2)
                                     / np.sum((wh - wh.mean()) ** 2)), 3)}
    out["fit_joint_sliced"] = {"mean_per_N": [round(float(m), 4) for m in sw],
                               "sd_per_N": [round(float(s), 4) for s in sw_sd],
                               "rmse_nhalf_shape": round(float(res_half), 5),
                               "rmse_log_shape": round(float(res_log), 5)}
    out["fit_note"] = ("marginals: W1 ~ a + b*N^{-1/2} (b/se reported; "
                       "a = discretisation floor); joint sliced-W1 reported "
                       "per run (mean +- SD over 8 runs), NOT pooled, and "
                       "compared against N^{-1/2} and log(1+N)/N^{1/2} "
                       "shapes (Theorem 3 joint rate)")

    # trajectory check at N=4000 (binned per-capita rates)
    p = build_mc_params(4000, np.repeat(np.arange(4),
                                        np.round(pi * 4000).astype(int)))
    n_bins = 20
    bin_edges = np.linspace(0, PARAMS["T"], n_bins + 1)
    psi1, psiC, psiH = responses()
    mark = logistic_mark(**PARAMS["mark"])
    t0 = time.time()
    sims = []
    for s in range(6):
        sim = EventSimulator(p, psi1, psiC, mark, psiH=psiH,
                             rng=np.random.default_rng(100 + s))
        sims.append(sim.run())
    mc_wall = (time.time() - t0) / 6
    mc_rates = np.zeros(n_bins)
    for res in sims:
        h, _ = np.histogram(res.t, bins=bin_edges)
        mc_rates += h / len(sims)
    mc_rates /= (PARAMS["T"] / n_bins)
    mc_rates /= p.N
    sol_traj = KineticSolver2D(build_spec(pi))
    macro_rates = sol_traj.event_rate_trajectory(n_bins)
    rel_diff = float(np.mean(np.abs(mc_rates - macro_rates)
                             / np.maximum(macro_rates, 1e-9)))
    out["trajectory"] = {
        "mc_rates": mc_rates.tolist(),
        "macro_rates": macro_rates.tolist(),
        "rel_mean_abs_diff": rel_diff,
    }

    out["wall_seconds"] = round(time.time() - t_start, 1)
    (EXP_DIR / "micro_macro_2d.json").write_text(json.dumps(out, indent=2))
    print(f"\nW1_C(N) = {a_c:.4f} + {b_c:.4f} N^-1/2")
    print(f"W1_H(N) = {a_h:.4f} + {b_h:.4f} N^-1/2")
    print(f"joint sliced: rmse(N^-1/2 shape)={res_half:.5f} vs "
          f"rmse(log shape)={res_log:.5f}")
    print(f"trajectory rel diff: {rel_diff:.4f}; "
          f"macro {macro_wall:.2f}s vs MC {mc_wall:.2f}s at N=4000")
    print("[ok] wrote experiments/micro_macro_2d.json")


if __name__ == "__main__":
    main()
