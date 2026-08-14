#!/usr/bin/env python
"""WP1: descriptive facts (research plan 8.3) -> experiments/facts.json + figures."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dengyunetwork import facts as F  # noqa: E402
from dengyunetwork import plotstyle as PS  # noqa: E402
from dengyunetwork.data import load_events, node_entry_times, sort_by_time  # noqa: E402

EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures"


def main() -> None:
    PS.apply()
    EXP_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    t_start = time.time()
    out: dict = {}

    df = sort_by_time(load_events())
    entry = node_entry_times(df)
    nodes = np.unique(np.concatenate([df["source"].values, df["target"].values]))
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()

    # --- 1. event & entry rates -------------------------------------------
    rates = F.monthly_rates(df)
    out["monthly_rates"] = rates.reset_index().assign(
        month=rates.index.astype(str)).to_dict(orient="records")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    ax.plot(np.arange(len(rates)), rates["events"].values, color=PS.C1)
    ax.set_title("Rating events per month")
    ax.set_xlabel("Month since first event")
    ax.set_ylabel("Events")
    ax = axes[1]
    ax.plot(np.arange(len(rates)), rates["new_users"].values, color=PS.C2)
    ax.set_title("New users per month")
    ax.set_xlabel("Month since first event")
    ax.set_ylabel("New users")
    PS.save(fig, FIG_DIR / "fig_f1_rates.png")

    # --- 2. out/in heterogeneity -----------------------------------------
    het = F.heterogeneity(df)
    out["heterogeneity"] = het
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, col, c in [("out-degree", "out", PS.C1), ("in-degree", "in", PS.C2)]:
        vals = het and None
        counts = df.groupby("source").size() if col == "out" else df.groupby("target").size()
        counts = counts.values
        ccdf = 1 - np.searchsorted(np.sort(counts), np.arange(1, counts.max() + 1)) / len(counts)
        ax.plot(np.arange(1, counts.max() + 1), ccdf, color=c, label=name, lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree k")
    ax.set_ylabel("P(degree >= k)")
    ax.set_title("Emitted / received rating heterogeneity")
    ax.legend()
    PS.save(fig, FIG_DIR / "fig_f2_heterogeneity.png")

    # --- 3. sign dynamics -------------------------------------------------
    signs = F.sign_series(df)
    out["sign_series"] = signs.reset_index().assign(
        month=signs.index.astype(str)).to_dict(orient="records")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(np.arange(len(signs)), signs["neg_share"].values, color=PS.C1)
    axes[0].set_title("Negative rating share by month")
    axes[0].set_xlabel("Month since first event")
    axes[0].set_ylabel("Share negative")
    axes[1].plot(np.arange(len(signs)), signs["mean_rating"].values, color=PS.C2)
    axes[1].set_title("Mean rating by month")
    axes[1].set_xlabel("Month since first event")
    axes[1].set_ylabel("Mean rating")
    PS.save(fig, FIG_DIR / "fig_f3_signs.png")

    # --- 4. capital distribution shape ------------------------------------
    shape_by_hl: dict[str, dict] = {}
    for hl, label in [(None, "no_decay"), (30, "hl30d"), (180, "hl180d"), (365, "hl365d")]:
        delta = F.LN2 / (hl * 86400.0) if hl else 0.0
        times, snap = F.capital_distribution_snapshots(df, nodes, entry, delta, "net")
        last = snap[-1]
        last = last[~np.isnan(last)]
        shape_by_hl[label] = F.distribution_shape(last)
    out["capital_shape_by_half_life"] = shape_by_hl

    # shape for the three capital definitions (hl=180d)
    cap_defs: dict[str, dict] = {}
    for cap_type in ["count", "net", "quality"]:
        delta = F.LN2 / (180 * 86400.0)
        times, snap = F.capital_distribution_snapshots(df, nodes, entry, delta, cap_type)
        vals = snap[-1]
        vals = vals[~np.isnan(vals)]
        cap_defs[cap_type] = F.distribution_shape(vals)
    out["capital_shape_by_definition"] = cap_defs

    # gini trajectory (count capital, hl=180d)
    delta = F.LN2 / (180 * 86400.0)
    times, snap_count = F.capital_distribution_snapshots(df, nodes, entry, delta, "count")
    gini_traj = []
    for k in range(snap_count.shape[0]):
        row = snap_count[k]
        gini_traj.append(F.gini(row[~np.isnan(row)]))
    out["gini_trajectory_count_hl180"] = {
        "month": list(range(1, len(gini_traj) + 1)), "gini": gini_traj}

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(np.arange(1, len(gini_traj) + 1), gini_traj, color=PS.C1)
    ax.set_title("Gini of count-capital over time (hl = 180d)")
    ax.set_xlabel("Snapshot month")
    ax.set_ylabel("Gini")
    PS.save(fig, FIG_DIR / "fig_f4_gini.png")

    # net-capital histogram at the last snapshot (hl=180d)
    delta = F.LN2 / (180 * 86400.0)
    _, snap_net = F.capital_distribution_snapshots(df, nodes, entry, delta, "net")
    net_vals = snap_net[-1]
    net_vals = net_vals[~np.isnan(net_vals)]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(net_vals, bins=60, color=PS.C1, edgecolor=PS.SURFACE, lw=0.5)
    ax.set_yscale("log")
    ax.set_title("Net-capital distribution at sample end (hl = 180d)")
    ax.set_xlabel("Decayed net capital")
    ax.set_ylabel("Users (log)")
    PS.save(fig, FIG_DIR / "fig_f5_capital_hist.png")

    # --- 5. target-selection probe (G1 identifiability) -------------------
    t0p = time.time()
    probe = F.run_g1_probe(df, entry, max_events=12000, n_controls=30, seed=0)
    probe["wall_seconds"] = round(time.time() - t0p, 1)
    out["target_selection_probe"] = probe

    feats = list(probe["coef"].keys())
    coef = np.array([probe["coef"][f] for f in feats])
    se = np.array([probe["se"][f] for f in feats])
    coef_neg = np.array([probe["coef_negative_control"][f] for f in feats])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    y = np.arange(len(feats))
    ax.errorbar(coef, y, xerr=1.96 * se, fmt="o", color=PS.C1, label="observed targets",
                capsize=3, ms=7)
    ax.errorbar(coef_neg, y, xerr=None, fmt="s", color=PS.C2, label="negative control",
                ms=6)
    ax.axvline(0, color=PS.MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Conditional-logit coefficient (being chosen as target)")
    ax.legend()
    ax.set_title("Target selection: covariates of being chosen")
    PS.save(fig, FIG_DIR / "fig_f6_target_probe.png")

    # --- 6. mixing between latent groups ----------------------------------
    train = df[df["timestamp"] <= df["timestamp"].quantile(0.7)]
    mix = F.mixing_matrix(train, k=5, seed=0)
    out["mixing"] = mix
    M = np.array(mix["mixing"])
    lift = np.array(mix["lift"])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    clipped = np.clip(lift, 0.2, 5.0)
    im = ax.imshow(clipped, cmap=PS.DIVERGING_CMAP,
                   norm=TwoSlopeNorm(vcenter=1.0, vmin=0.2, vmax=5.0), aspect="auto")
    for a in range(5):
        for b in range(5):
            ax.text(b, a, f"{lift[a, b]:.2f}", ha="center", va="center",
                    fontsize=8, color=PS.INK)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xlabel("Target latent group")
    ax.set_ylabel("Source latent group")
    ax.set_title("Mixing lift vs random (training period)")
    fig.colorbar(im, ax=ax, label="lift = observed / expected")
    PS.save(fig, FIG_DIR / "fig_f7_mixing.png")

    # --- 7. once-per-pair constraint vs risk set --------------------------
    n_events = len(df)
    used = np.zeros(n_events)
    n_active = np.zeros(n_events)
    act_count = 0
    seen = set()
    present = np.zeros(len(nodes))
    pos = {nd: k for k, nd in enumerate(nodes)}
    for e, row in enumerate(sort_by_time(df).itertuples(index=False)):
        i, j = pos[row.source], pos[row.target]
        if present[i] == 0:
            present[i] = 1
            act_count += 1
        if present[j] == 0:
            present[j] = 1
            act_count += 1
        seen.add((row.source, row.target))
        used[e] = len(seen)
        n_active[e] = act_count
    frac = used / np.maximum(n_active * (n_active - 1), 1)
    out["once_per_pair"] = {
        "final_active": int(n_active[-1]),
        "final_used_pairs": int(used[-1]),
        "risk_set_without_constraint": int(n_active[-1] * (n_active[-1] - 1)),
        "final_constraint_share": float(frac[-1]),
        "max_constraint_share": float(frac.max()),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(np.arange(n_events), n_active, color=PS.C1)
    axes[0].set_title("Active users (risk set population)")
    axes[0].set_xlabel("Event index")
    axes[0].set_ylabel("Active users")
    axes[1].plot(np.arange(n_events), frac, color=PS.C2)
    axes[1].set_title("Used directed pairs / candidate pairs")
    axes[1].set_xlabel("Event index")
    axes[1].set_ylabel("Share of risk set consumed")
    PS.save(fig, FIG_DIR / "fig_f8_once_per_pair.png")

    # --- write -------------------------------------------------------------
    out["wall_seconds"] = round(time.time() - t_start, 1)
    (EXP_DIR / "facts.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in
                      ("monthly_rates", "sign_series", "mixing")}, indent=2, default=str))
    print(f"\n[ok] wrote experiments/facts.json ({out['wall_seconds']}s)")


if __name__ == "__main__":
    main()
