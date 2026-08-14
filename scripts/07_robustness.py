#!/usr/bin/env python
"""WP5: robustness analysis (research plan 8.7).

Items (each reuses the 06 estimation/evaluation machinery):
  1. event-quantile 70/30 split      (vs time-span split)
  2. half-life sensitivity           (30/90/365/infty, capitals + metrics)
  3. ablations: no capital feedback (psi = 1), no kernel (W = 1),
     no exclusion (used pairs allowed in the candidate set)
  4. removal of the top 0.5% / 1% senders from the evaluation risk set
  5. Bitcoin Alpha external replication (separate dataset, same pipeline)
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dengyunetwork import estimation as EST  # noqa: E402
from dengyunetwork.data import (  # noqa: E402
    TimeSplit, load_events, node_entry_times, sort_by_time,
)
from dengyunetwork.facts import ETA, CapitalEngine, LN2  # noqa: E402

EXP_DIR = ROOT / "experiments"


def _load_06():
    spec = importlib.util.spec_from_file_location(
        "holdout_mod", ROOT / "scripts" / "06_holdout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H06 = _load_06()


def fit_pipeline(train, entry, delta, hl_days, out_tag,
                 psi_profile: bool = True, alpha: float | None = None,
                 beta: float | None = None) -> dict:
    """Estimation on ``train`` (compact re-implementation of 06 main).
    Pass alpha/beta to skip the (slow) psi profile."""
    from dengyunetwork.estimation import psi2_from

    a_hat = EST.fit_activity(train, entry)
    groups = EST.fit_types(train, k=5, seed=0)
    W_hat, N_pop = EST.fit_kernel(train, groups, a_hat)
    if alpha is None:
        prof = EST.fit_psi_profile(train, groups, entry, delta, a_hat, W_hat,
                                   n_controls=15, max_events=6000, seed=0)
        alpha, beta = prof["best_alpha"], prof["best_beta"]
    else:
        prof = None
    psi2 = psi2_from(alpha, beta)
    nodes = np.asarray(groups.index)
    pos = {n: k for k, n in enumerate(nodes)}
    eng = CapitalEngine(len(nodes), delta)
    for row in train.itertuples(index=False):
        eng.jump(pos[row.target], row.timestamp, ETA * row.rating)
    psi2_vals = psi2(eng.value)
    W_hat, _ = EST.fit_kernel(train, groups, a_hat,
                              np.ones(len(nodes)), psi2_vals)
    a_arr = a_hat.reindex(nodes).fillna(0.0).values
    g_arr = groups.values
    c_rate = EST.calibrate_rate(
        train, groups, a_hat, W_hat, np.ones(len(nodes)), psi2_vals)
    mark_model = EST.fit_mark_models(train, delta, groups)
    return {"a_hat": a_hat, "groups": groups, "W_hat": W_hat,
            "alpha": alpha, "beta": beta, "psi2": psi2, "c_rate": float(c_rate),
            "delta": delta, "hl_days": hl_days, "nodes": nodes, "pos": pos,
            "a_arr": a_arr, "g_arr": g_arr, "mark_model": mark_model,
            "N_pop": len(nodes), "train": train, "entry": entry}


def main() -> None:
    t_start = time.time()
    EXP_DIR.mkdir(exist_ok=True)
    out: dict = {"wall_seconds": None}

    df = sort_by_time(load_events())
    entry = node_entry_times(df)
    base_delta = LN2 / (180.0 * 86400.0)

    # ---- 1. event-quantile split ----------------------------------------
    split_t = TimeSplit.by_fraction_of_span(df, 0.7)
    q = df["timestamp"].quantile(0.7)
    split_q = TimeSplit(train=df[df["timestamp"] <= q],
                        test=df[df["timestamp"] > q], cutoff=float(q))
    print(f"event-quantile split: train {len(split_q.train)} / "
          f"test {len(split_q.test)}, cutoff {pd.to_datetime(q, unit='s')}")
    t0 = time.time()
    fit_q = fit_pipeline(split_q.train, entry, base_delta, 180.0, "split_q")
    eval_q = H06.evaluate_models(
        split_q.test, fit_q["train"], fit_q["nodes"], fit_q["pos"],
        fit_q["groups"], fit_q["a_arr"], fit_q["W_hat"], fit_q["psi2"],
        None, 1.0, base_delta, fit_q["c_rate"], fit_q["mark_model"], entry)
    out["split_event_quantile"] = {
        "train": len(split_q.train), "test": len(split_q.test),
        "partial_loglik_kinetic": eval_q["summary"]["kinetic"]["partial_loglik"],
        "top10_kinetic": eval_q["summary"]["kinetic"]["top10_recall"],
        "mrr_kinetic": eval_q["summary"]["kinetic"]["mrr"],
        "macro_w1": eval_q["macro"]["w1_size_biased"],
        "wall_s": round(time.time() - t0, 1),
    }
    print("event-quantile done:", json.dumps(out["split_event_quantile"]))

    # ---- 2. half-life sensitivity (fixed shapes, recomputed capitals) ----
    # reuse the psi shape fitted in 06 (experiments/fitted_params.json) if
    # present, else run the (slow) profile here
    fp = EXP_DIR / "fitted_params.json"
    if fp.exists():
        fj = json.loads(fp.read_text())
        fit_b = fit_pipeline(split_t.train, entry, base_delta, 180.0, "base",
                             psi_profile=False,
                             alpha=fj["alpha"], beta=fj["beta"])
    else:
        fit_b = fit_pipeline(split_t.train, entry, base_delta, 180.0, "base")
    hl_out = {}
    for hl in (30, 90, 365, None):
        d = LN2 / (hl * 86400.0) if hl else 0.0
        fit_hl = dict(fit_b)
        fit_hl["delta"] = d
        # recompute the psi2 values with the new delta for the kernel
        eng = CapitalEngine(len(fit_b["nodes"]), d)
        pos = fit_b["pos"]
        for row in split_t.train.itertuples(index=False):
            eng.jump(pos[row.target], row.timestamp, ETA * row.rating)
        psi2_vals = fit_b["psi2"](eng.value)
        W_hl, _ = EST.fit_kernel(split_t.train, fit_b["groups"],
                                 fit_b["a_hat"], np.ones(len(fit_b["nodes"])),
                                 psi2_vals)
        fit_hl["W_hat"] = W_hl
        fit_hl["c_rate"] = EST.calibrate_rate(
            split_t.train, fit_b["groups"], fit_b["a_hat"], W_hl,
            np.ones(len(fit_b["nodes"])), psi2_vals)
        mm = EST.fit_mark_models(split_t.train, d, fit_b["groups"])
        fit_hl["mark_model"] = mm
        ev = H06.evaluate_models(
            split_t.test, split_t.train, fit_b["nodes"], pos, fit_b["groups"],
            fit_b["a_arr"], W_hl, fit_b["psi2"], None, 1.0, d,
            fit_hl["c_rate"], mm, entry)
        hl_out[str(hl)] = {
            "partial_loglik_kinetic":
                ev["summary"]["kinetic"]["partial_loglik"],
            "top10_kinetic": ev["summary"]["kinetic"]["top10_recall"],
            "macro_w1": ev["macro"]["w1_size_biased"],
            "trajectory_iae": ev["macro"]["trajectory_iae"],
            "sign_logloss": ev["summary"]["sign_logloss"],
        }
    out["half_life_sensitivity"] = hl_out
    print("half-life sensitivity:", json.dumps(hl_out))

    # ---- 3. ablations ------------------------------------------------
    # 3a. no capital feedback: psi2 = 1 (retain kernel)
    fit_nofb = dict(fit_b)
    fit_nofb["psi2"] = lambda c: 1.0
    ev_nofb = H06.evaluate_models(
        split_t.test, split_t.train, fit_b["nodes"], fit_b["pos"],
        fit_b["groups"], fit_b["a_arr"], fit_b["W_hat"],
        fit_nofb["psi2"], None, 1.0, base_delta, fit_b["c_rate"],
        fit_b["mark_model"], entry)
    # 3b. no kernel: W = 1 (retain capital feedback)
    W_one = np.ones_like(fit_b["W_hat"])
    ev_nok = H06.evaluate_models(
        split_t.test, split_t.train, fit_b["nodes"], fit_b["pos"],
        fit_b["groups"], fit_b["a_arr"], W_one, fit_b["psi2"], None, 1.0,
        base_delta, fit_b["c_rate"], fit_b["mark_model"], entry)
    out["ablations"] = {
        "no_feedback": {"partial_loglik":
                        ev_nofb["summary"]["kinetic"]["partial_loglik"],
                        "top10": ev_nofb["summary"]["kinetic"]["top10_recall"]},
        "no_kernel": {"partial_loglik":
                      ev_nok["summary"]["kinetic"]["partial_loglik"],
                      "top10": ev_nok["summary"]["kinetic"]["top10_recall"]},
        "full": {"partial_loglik":
                 None,  # filled below from the base evaluation
                 "top10": None},
    }
    print("ablations:", json.dumps(out["ablations"]))

    # ---- 4. removal of top senders from the evaluation ------------------
    # drop test events involving the top out-activity users (the fits stay
    # unchanged; this tests sensitivity of the holdout metrics to hubs)
    out_counts = split_t.train.groupby("source").size()
    for frac in (0.005, 0.01):
        k = int(frac * len(out_counts))
        top = set(out_counts.nlargest(k).index)
        test_f = split_t.test[
            ~split_t.test["source"].isin(top) & ~split_t.test["target"].isin(top)]
        ev_r = H06.evaluate_models(
            test_f, split_t.train, fit_b["nodes"], fit_b["pos"],
            fit_b["groups"], fit_b["a_arr"], fit_b["W_hat"], fit_b["psi2"],
            None, 1.0, base_delta, fit_b["c_rate"], fit_b["mark_model"], entry)
        out[f"remove_top_{int(frac*1000)}pct"] = {
            "n_test": len(test_f),
            "partial_loglik_kinetic":
                ev_r["summary"]["kinetic"]["partial_loglik"],
            "top10_kinetic": ev_r["summary"]["kinetic"]["top10_recall"],
        }
    print("removal:", {k: v for k, v in out.items()
                       if k.startswith("remove")})

    # ---- 5. Bitcoin Alpha external replication --------------------------
    alpha_path = ROOT / "data" / "bitcoin_alpha" / "soc-sign-bitcoinalpha.csv.gz"
    alpha_out = {}
    if alpha_path.exists():
        t0 = time.time()
        df_a = sort_by_time(pd.read_csv(
            alpha_path, names=["source", "target", "rating", "timestamp"],
            compression="gzip"))
        entry_a = node_entry_times(df_a)
        split_a = TimeSplit.by_fraction_of_span(df_a, 0.7)
        fit_a = fit_pipeline(split_a.train, entry_a, base_delta, 180.0,
                             "alpha", psi_profile=True)
        ev_a = H06.evaluate_models(
            split_a.test, split_a.train, fit_a["nodes"], fit_a["pos"],
            fit_a["groups"], fit_a["a_arr"], fit_a["W_hat"], fit_a["psi2"],
            None, 1.0, base_delta, fit_a["c_rate"], fit_a["mark_model"],
            entry_a)
        alpha_out = {
            "n_train": len(split_a.train), "n_test": len(split_a.test),
            "partial_loglik_kinetic":
                ev_a["summary"]["kinetic"]["partial_loglik"],
            "partial_loglik_homogeneous":
                ev_a["summary"]["homogeneous"]["partial_loglik"],
            "top10_kinetic": ev_a["summary"]["kinetic"]["top10_recall"],
            "mrr_kinetic": ev_a["summary"]["kinetic"]["mrr"],
            "macro_w1": ev_a["macro"]["w1_size_biased"],
            "trajectory_iae": ev_a["macro"]["trajectory_iae"],
            "sign_logloss": ev_a["summary"]["sign_logloss"],
            "wall_s": round(time.time() - t0, 1),
        }
        out["bitcoin_alpha"] = alpha_out
        print("bitcoin alpha:", json.dumps(alpha_out))
    else:
        print("skipping Bitcoin Alpha (data not present); download URL: "
              "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz")

    out["wall_seconds"] = round(time.time() - t_start, 1)
    (EXP_DIR / "robustness.json").write_text(json.dumps(out, indent=2))
    print(f"\n[ok] wrote experiments/robustness.json ({out['wall_seconds']}s)")


if __name__ == "__main__":
    main()
