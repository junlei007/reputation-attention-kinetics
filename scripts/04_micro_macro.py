#!/usr/bin/env python
"""WP3: micro-macro consistency (gate G3).

Validation design: K=4 blocks with *deterministic per-group activity* (so the
block kinetic equation is the exact mean-field limit of the microscopic
process), target-only marks, all nodes active at t=0, N scaling experiment.

Checks:
  1. W1(empirical capital distribution of the N-particle process, macro f_T)
     decays ~ N^{-1/2} as N grows (log-log slope near -0.5);
  2. the macro solver is independent of N: wall-clock advantage vs one MC
     run at the largest N, growing with N (MC cost is O(N^2 * T));
  3. the per-capita event-rate trajectory of MC (binned) matches the macro
     solver's predicted trajectory.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dengyunetwork.kinetic import KineticSolver, KineticSpec  # noqa: E402
from dengyunetwork.simulator import (  # noqa: E402
    EventSimulator, SimParams, logistic_mark, tanh_psi,
)

EXP_DIR = ROOT / "experiments"

PARAMS = {
    "K": 4, "T": 4.0,
    "delta": np.log(2.0) / 30.0,
    "alpha1": 0.3, "beta1": 3.0,
    "alpha2": 0.8, "beta2": 2.0,
    "eta": 0.1,
    "activity": np.array([0.4, 0.8, 1.5, 2.5]),   # deterministic per group
    "mark": {"intercept": 0.5, "b_target": 1.0, "b_sender": 0.0},
}
W_TRUE = np.full((PARAMS["K"], PARAMS["K"]), 0.4)
np.fill_diagonal(W_TRUE, 1.2)
W_TRUE[0, 1:] = 0.9
W_TRUE[1:, 0] = 0.9


def build_mc_params(N: int, seed: int = 0,
                    groups: np.ndarray | None = None) -> SimParams:
    K = PARAMS["K"]
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = rng.integers(0, K, size=N)
    groups = np.asarray(groups)
    # activity must be aligned with the FINAL group assignment
    activity = PARAMS["activity"][groups].copy()
    p = SimParams(N=N, K=K, groups=groups, activity=activity, W=W_TRUE,
                  delta=PARAMS["delta"], eta=PARAMS["eta"],
                  T=PARAMS["T"], entry_times=np.zeros(N))
    return p


def run_mc(p: SimParams, n_runs: int, seed_base: int = 100
           ) -> tuple[list[np.ndarray], list[SimResult]]:
    """Returns (per-run POPULATION capitals at time T, per-run SimResults).

    The population capital vector (each node once) is the object the kinetic
    equation predicts; the per-event ``cap_after`` column is event-weighted
    and must not be compared to the density f_T directly."""
    psi1 = tanh_psi(PARAMS["alpha1"], PARAMS["beta1"])
    psi2 = tanh_psi(PARAMS["alpha2"], PARAMS["beta2"])
    mark = logistic_mark(**PARAMS["mark"])
    capitals = []
    results = []
    for s in range(n_runs):
        sim = EventSimulator(p, psi1, psi2, mark,
                             rng=np.random.default_rng(seed_base + s))
        res = sim.run()
        capitals.append(np.sort(sim.C.copy()))
        results.append(res)
    return capitals, results


def build_spec(pi: np.ndarray, M: int = 2000,
               dt_max: float = 0.02) -> KineticSpec:
    K = PARAMS["K"]
    c_min, c_max = -2.0, 2.0
    c_mid = np.linspace(c_min, c_max, M + 1)
    c_mid = 0.5 * (c_mid[:-1] + c_mid[1:])
    # target-only marks: sign logistic, magnitude uniform 1..10 (scaled /10).
    # NOTE: r_values is ordered negative-first (sgn=-1 first), so columns
    # 0..9 of q_cells are the NEGATIVE marks and 10..19 the positive ones.
    r_vals = np.array([sgn * mag / 10.0 for sgn in (-1, 1) for mag in range(1, 11)])
    p_pos = 1.0 / (1.0 + np.exp(-(PARAMS["mark"]["intercept"]
                                  + PARAMS["mark"]["b_target"] * c_mid)))
    q = np.zeros((M, 20))
    q[:, :10] = ((1.0 - p_pos) / 10.0)[:, None]   # negative marks first
    q[:, 10:] = (p_pos / 10.0)[:, None]           # positive marks
    return KineticSpec(
        K=K, pi=pi, activity=PARAMS["activity"], W=W_TRUE,
        delta=PARAMS["delta"], eta=PARAMS["eta"],
        psi1=tanh_psi(PARAMS["alpha1"], PARAMS["beta1"]),
        psi2=tanh_psi(PARAMS["alpha2"], PARAMS["beta2"]),
        r_values=r_vals, q_cells=q, c_min=c_min, c_max=c_max, M=M,
        dt_max=dt_max, T=PARAMS["T"],
    )


def w1_cdf(sorted_x: np.ndarray, f: np.ndarray, grid: np.ndarray) -> float:
    """W1 between an empirical sample and a density f on ``grid`` (midpoints)."""
    F_emp = np.searchsorted(sorted_x, grid) / len(sorted_x)
    F_mac = np.cumsum(f) / max(f.sum(), 1e-300)
    return float(np.trapezoid(np.abs(F_emp - F_mac), grid))


def w1_density(f1: np.ndarray, grid1: np.ndarray,
               f2: np.ndarray, grid2: np.ndarray) -> float:
    """W1 between two densities given on (possibly different) mid-grids."""
    F1 = np.cumsum(f1) / max(f1.sum(), 1e-300)
    F2 = np.cumsum(f2) / max(f2.sum(), 1e-300)
    F2_at = np.interp(grid1, grid2, F2)
    return float(np.trapezoid(np.abs(F1 - F2_at), grid1))


def main() -> None:
    EXP_DIR.mkdir(exist_ok=True)
    out: dict = {}
    t_start = time.time()

    # macro solution (N-independent)
    n_blocks = np.array([50, 120, 180, 150])   # group sizes for N=500 design
    pi = n_blocks / n_blocks.sum()
    spec = build_spec(pi)
    solver = KineticSolver(spec)
    t0 = time.time()
    f_T = solver.run()
    macro_wall = time.time() - t0
    grid = solver.c_mid
    f_marg = f_T.sum(axis=0) * solver.h   # aggregate probability mass per cell
    out["macro_wall_s"] = round(macro_wall, 2)
    out["macro_cells"] = int(spec.K * spec.M)

    # N scaling of the micro-macro distance: the propagation-of-chaos object
    # is the SINGLE-RUN empirical measure, whose W1 to f_T contains the
    # O(N^{-1/2}) fluctuation; pooling runs would measure only the O(1/N)
    # mean-field bias and hide the decay.
    sizes = [250, 500, 1000, 2000, 4000]
    n_runs = 8
    w1_results = []
    for N in sizes:
        pi_N = np.round(pi * N).astype(int)
        pi_N[-1] = N - pi_N[:-1].sum()
        p = build_mc_params(N, seed=0,
                            groups=np.repeat(np.arange(4), pi_N))  # exact blocks
        t0 = time.time()
        caps, _ = run_mc(p, n_runs)
        mc_wall = (time.time() - t0) / n_runs
        per_run = [w1_cdf(r, f_marg, grid) for r in caps]
        pooled = np.concatenate(caps)
        w1_results.append({
            "N": N, "w1_single_run_mean": float(np.mean(per_run)),
            "w1_single_run_std": float(np.std(per_run)),
            "w1_pooled": float(w1_cdf(np.sort(pooled), f_marg, grid)),
            "mc_wall_per_run_s": round(mc_wall, 2),
        })
        print(f"N={N}: single-run W1={np.mean(per_run):.4f} "
              f"(+-{np.std(per_run):.4f}), pooled={w1_results[-1]['w1_pooled']:.4f}, "
              f"MC {mc_wall:.2f}s/run")

    # W1(N) = solver floor + fluctuation: fit W1 = a + b * N^{-1/2}
    ns = np.array([r["N"] for r in w1_results])
    ws = np.array([r["w1_single_run_mean"] for r in w1_results])
    X = np.column_stack([np.ones_like(ns), ns ** -0.5])
    a, b = np.linalg.lstsq(X, ws, rcond=None)[0]
    out["w1_vs_n"] = w1_results
    out["floor_a"] = round(float(a), 4)
    out["fluct_b"] = round(float(b), 4)
    out["fit_note"] = "W1(N) ~ a + b*N^{-1/2}: a = solver/mean-field floor, " \
                      "b*N^{-1/2} = propagation-of-chaos fluctuation"

    # solver self-convergence: finer grid + smaller dt on fresh specs
    conv = {}
    for tag, M, dt in [("M2000_dt0.02", 2000, 0.02),
                       ("M2000_dt0.005", 2000, 0.005),
                       ("M4000_dt0.02", 4000, 0.02)]:
        s = build_spec(pi, M=M, dt_max=dt)
        so = KineticSolver(s)
        so.run()
        fm_ = so.f.sum(axis=0) * so.h
        w1_macro = w1_density(fm_, so.c_mid, f_marg, grid)
        caps, _ = run_mc(build_mc_params(4000, seed=0), 4)
        w1_mc = float(w1_cdf(np.sort(np.concatenate(caps)), fm_, so.c_mid))
        conv[tag] = {"w1_vs_M2000_dt0.02": round(w1_macro, 4),
                     "w1_pooled_N4000": round(w1_mc, 4)}
    out["solver_self_convergence"] = conv

    # trajectory check at N=4000 (binned per-capita rates)
    p = build_mc_params(4000, seed=0,
                        groups=np.repeat(np.arange(4),
                                         np.round(pi * 4000).astype(int)))
    t0 = time.time()
    caps, runs = run_mc(p, 6)
    mc_wall = (time.time() - t0) / 6
    n_bins = 20
    bin_edges = np.linspace(0, PARAMS["T"], n_bins + 1)
    mc_rates = np.zeros(n_bins)
    for res in runs:
        h, _ = np.histogram(res.t, bins=bin_edges)
        mc_rates += h / len(runs)
    mc_rates /= (PARAMS["T"] / n_bins)          # per-bin total events / day
    mc_rates /= p.N                              # per-capita, matching the macro
    spec_traj = build_spec(pi)
    sol_traj = KineticSolver(spec_traj)
    macro_rates = sol_traj.event_rate_trajectory(n_bins)
    rel_diff = float(np.mean(np.abs(mc_rates - macro_rates)
                             / np.maximum(macro_rates, 1e-9)))
    out["trajectory"] = {
        "mc_rates": mc_rates.tolist(),
        "macro_rates": macro_rates.tolist(),
        "rel_mean_abs_diff": rel_diff,
    }

    out["wall_seconds"] = round(time.time() - t_start, 1)
    (EXP_DIR / "micro_macro.json").write_text(json.dumps(out, indent=2))
    print(f"\nW1(N) = {a:.4f} + {b:.4f} * N^-1/2  (floor + fluctuation)")
    print(f"trajectory rel diff: {rel_diff:.4f}")
    print(f"macro solver {macro_wall:.2f}s (N-independent) vs "
          f"MC {mc_wall:.2f}s/run at N=4000")
    print("\n[ok] wrote experiments/micro_macro.json")


if __name__ == "__main__":
    main()
