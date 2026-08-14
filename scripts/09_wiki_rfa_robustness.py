#!/usr/bin/env python
"""Wiki-RfA robustness battery for the supplement.

Covers the audit-requested checklist for the RfA negative result:
  1. coefficient stability across 4 random seeds (with standard errors);
  2. recency definition: capped recency (30 / 90 / 180 days);
  3. first-candidacy vs repeat-candidacy stratification (zero vs nonzero
     pre-window capital of the case target);
  4. dropping single-vote windows from the risk set;
  5. risk-set definition: window clustering gap 60 -> 90 days;
  6. capital half-life sensitivity (90 / 180 / 365 days): target choice,
     sign prediction, and election AUC;
  7. three-setting comparison table (OTC G1 probe, Alpha G1 probe, RfA
     aligned probe);
  8. arrival-rate shape tests (RfA; OTC/Alpha from the holdout runs).

All runs reuse the fixed pipeline of 08_wiki_rfa.py (no pooling with
Bitcoin; RfA fitted separately).
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

spec = importlib.util.spec_from_file_location("rfa", ROOT / "scripts" / "08_wiki_rfa.py")
rfa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfa)

from dengyunetwork.facts import run_g1_probe  # noqa: E402
from dengyunetwork.data import load_events, node_entry_times  # noqa: E402

EXP_DIR = ROOT / "experiments"
SEEDS = [0, 1, 2, 3]
HALF_LIVES = [90.0, 180.0, 365.0]
RECENCY_CAPS = [30.0, 90.0, 180.0]

FEATS = ["net_cap", "count_cap", "log_indegree", "age_days",
         "log_recency", "ever_sent"]


def load_rfa() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ok = rfa.parse_rfa().dropna(subset=["src", "tgt", "timestamp"])
    ok = ok.sort_values("timestamp").reset_index(drop=True)
    ok["rating_n"] = ok["rating"].astype(float)
    # net capital over the full stream (hl=180d baseline), as in 08 main
    delta = rfa.LN2 / (180.0 * 86400.0)
    nodes = np.unique(np.concatenate([ok["src"].values, ok["tgt"].values]))
    pos = {n: k for k, n in enumerate(nodes)}
    eng = rfa.CapitalEngine(len(nodes), delta)
    last_t = np.full(len(nodes), -np.inf)
    caps = np.zeros(len(ok))
    for e, row in enumerate(ok.itertuples(index=False)):
        j = pos[row.tgt]
        if row.timestamp > last_t[j]:
            eng.value[j] *= np.exp(-delta * (row.timestamp - last_t[j]))
            last_t[j] = row.timestamp
        caps[e] = eng.value[j]
        eng.value[j] += rfa.ETA * row.rating_n
    ok["net_cap"] = caps
    t0, t1 = ok["timestamp"].min(), ok["timestamp"].max()
    cutoff = t0 + 0.7 * (t1 - t0)
    test = ok[ok["timestamp"] > cutoff]
    return ok, test, {"nodes": nodes, "pos": pos, "cutoff": float(cutoff)}


def fit_conditional(cc: pd.DataFrame, tag: str) -> dict:
    from statsmodels.discrete.conditional_models import ConditionalLogit

    if cc["stratum"].nunique() < 20:
        return {"tag": tag, "n_strata": 0, "note": "too few strata"}
    try:
        m = ConditionalLogit(endog=cc["case"], exog=cc[FEATS],
                             groups=cc["stratum"])
        res = m.fit(disp=False, maxiter=300)
    except Exception as exc:  # degenerate stratum subset
        return {"tag": tag, "n_strata": int(cc["stratum"].nunique()),
                "note": f"fit failed: {type(exc).__name__}"}
    conv = getattr(res, "mle_retvals", None)
    converged = bool(conv.get("converged", True)) if isinstance(conv, dict) \
        else True
    return {"tag": tag, "n_strata": int(cc["stratum"].nunique()),
            "converged": converged,
            "coef": {f: float(res.params[f]) for f in FEATS},
            "se": {f: float(res.bse[f]) for f in FEATS},
            "pvalue": {f: float(res.pvalues[f]) for f in FEATS}}


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(exist_ok=True)
    out: dict = {"pipeline": "09_wiki_rfa_robustness",
                 "note": "RfA fitted separately; no pooling with Bitcoin"}

    ok, test, ctx = load_rfa()
    nodes, pos = ctx["nodes"], ctx["pos"]
    windows = rfa.build_windows(ok)
    delta180 = rfa.LN2 / (180.0 * 86400.0)

    # ---------------------------------------------------------------- 1. seeds
    out["seeds"] = []
    for s in SEEDS:
        r = rfa.window_target_choice(ok, test, windows, nodes, pos,
                                     delta180, seed=s)
        out["seeds"].append({"seed": s, "n_strata": r["n_strata"],
                             "converged": r["converged"],
                             "coef": r["coef"], "se": r["se"]})
    print("seeds done")

    # ---------------------------------------------------- 2. recency caps
    out["recency_caps"] = []
    for cap in RECENCY_CAPS:
        r = rfa.window_target_choice(ok, test, windows, nodes, pos,
                                     delta180, seed=0,
                                     recency_cap_days=cap)
        out["recency_caps"].append({"cap_days": cap, "n_strata": r["n_strata"],
                                    "coef": r["coef"], "se": r["se"]})
    print("recency caps done")

    # -------------------------------------- 3. first vs repeat candidacy
    _, cc = rfa.window_target_choice(ok, test, windows, nodes, pos,
                                     delta180, seed=0, return_strata=True)
    # the case target's net capital at the first in-window event IS its
    # pre-window capital (recorded before the vote is applied)
    cc["prior_cap"] = np.where(cc["case"] == 1, cc["net_cap"], np.nan)
    prior = cc.groupby("stratum")["prior_cap"].transform("first")
    cc_first = cc[prior == 0.0].copy()
    cc_rep = cc[prior != 0.0].copy()
    out["stratify_prior_capital"] = {
        "first_candidacy": fit_conditional(cc_first, "first (zero prior cap)"),
        "repeat_candidacy": fit_conditional(cc_rep, "repeat (nonzero prior cap)"),
        "n_first_strata": int(cc_first["stratum"].nunique()),
        "n_repeat_strata": int(cc_rep["stratum"].nunique()),
    }
    print("stratification done")

    # ------------------------------------------------- 4. drop single-vote windows
    r = rfa.window_target_choice(ok, test, windows, nodes, pos, delta180,
                                 seed=0, min_window_votes=2)
    out["drop_single_vote_windows"] = {"n_strata": r["n_strata"],
                                       "coef": r["coef"], "se": r["se"]}
    print("single-vote drop done")

    # ------------------------------------------------------ 5. gap 90 days
    windows90 = rfa.build_windows(ok, gap_days=90.0)
    r90 = rfa.window_target_choice(ok, test, windows90, nodes, pos,
                                   delta180, seed=0)
    e90 = rfa.election_validation(ok, windows90, delta180)
    out["gap_90_days"] = {
        "n_windows": int(len(windows90)),
        "median_window_days": float(((windows90["t_end"] - windows90["t_start"])
                                     / 86400.0).median()),
        "target_choice": {"n_strata": r90["n_strata"], "coef": r90["coef"],
                          "se": r90["se"]},
        "election": {k: e90[k] for k in
                     ["n_windows", "auc_pre_window_capital",
                      "auc_pre_window_capital_nonzero"]},
    }
    print("gap 90 done")

    # ------------------------------------------------------- 6. half-life
    out["half_life"] = []
    for hl in HALF_LIVES:
        d = rfa.LN2 / (hl * 86400.0)
        tc = rfa.window_target_choice(ok, test, windows, nodes, pos, d, seed=0)
        ev = rfa.election_validation(ok, windows, d)
        # sign model on the same time split as the pipeline
        tr = ok.iloc[: int(0.7 * len(ok))]
        te = ok.iloc[int(0.7 * len(ok)):]
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=3000).fit(
            tr[["net_cap"]].values, (tr["rating_n"].values > 0).astype(int))
        p = lr.predict_proba(te[["net_cap"]].values)[:, 1]
        y_te = (te["rating_n"].values > 0).astype(float)
        ll = float(np.mean(y_te * np.log(np.clip(p, 1e-9, 1))
                           + (1 - y_te) * np.log(np.clip(1 - p, 1e-9, 1))))
        base = float(y_te.mean())
        ll0 = base * np.log(base) + (1 - base) * np.log(1 - base)
        out["half_life"].append({
            "hl_days": hl,
            "target_choice": {"n_strata": tc["n_strata"], "coef": tc["coef"]},
            "election_auc": ev["auc_pre_window_capital"],
            "sign_logloss": {"model": round(-ll, 4), "baseline": round(-ll0, 4)},
        })
    print("half-life done")

    # ------------------------------------------------ 7. three settings
    # OTC G1 probe (already in experiments/facts.json): read and echo
    facts = json.loads((EXP_DIR / "facts.json").read_text())
    g1 = facts.get("target_selection_probe", {})
    # Alpha G1 probe (run here, separately fitted)
    alpha_df = load_events(str(ROOT / "data" / "bitcoin_alpha"
                               / "soc-sign-bitcoinalpha.csv.gz"))
    alpha_entry = node_entry_times(alpha_df)
    alpha_probe = run_g1_probe(alpha_df, alpha_entry,
                               max_events=12000, n_controls=30, seed=0)
    out["three_settings"] = {
        "otc_g1": {"n_events": g1.get("n_events"),
                   "coef": g1.get("coef"), "se": g1.get("se")},
        "alpha_g1": {"n_events": alpha_probe["n_events"],
                     "coef": alpha_probe["coef"],
                     "se": alpha_probe["se"],
                     "pvalue": alpha_probe["pvalue"]},
        "rfa_aligned": {"n_strata": out["seeds"][0]["n_strata"],
                        "coef": out["seeds"][0]["coef"],
                        "se": out["seeds"][0]["se"]},
    }
    print("three settings done")

    # ---------------------------------------------- 8. arrival-rate shape
    traj = rfa.arrival_trajectory(ok, ok[ok["timestamp"] <= ctx["cutoff"]],
                                  test, nodes, pos, delta180, ctx["cutoff"],
                                  rfa.ETA)
    out["arrival_shape"] = {
        "rfa_shape_corr": traj["shape_corr"],
        "rfa_rel_mean_abs_diff": traj["rel_mean_abs_diff_after_scale"],
        "note": "OTC/Alpha arrival shape: reported in experiments/"
                "holdout.json and robustness.json (macro trajectory IAE "
                "16.3, sixfold decline missed); both fail like RfA",
    }

    out["wall_seconds"] = round(time.time() - t0, 1)
    (EXP_DIR / "wiki_rfa_robustness.json").write_text(
        json.dumps(out, indent=2))
    print(f"\n[ok] wrote experiments/wiki_rfa_robustness.json "
          f"({out['wall_seconds']}s)")


if __name__ == "__main__":
    main()
