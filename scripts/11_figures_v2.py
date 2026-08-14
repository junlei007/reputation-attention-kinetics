#!/usr/bin/env python
"""Two main-text figures (v2, acceptance-round fixes).

fig_theory_error:  three short process boxes (micro events -> capital
  update -> kinetic limit) plus one FULL-WIDTH error-bound panel; one
  formula and one keyword line per box; no in-figure literature
  comparison; the B_i == 1 assumption is stated in the panel.

fig_three_settings:  predictor-standardised conditional-logit coefficients
  (beta * SD of the predictor within each dataset's analysis strata) with
  model-based 95% conditional-logit intervals; marker shapes distinguish
  datasets (circle / square / triangle); the right panel keeps only the
  three cross-dataset conclusions, with no mixing of standardised and raw
  coefficients.

Both figures are designed for a final width of 5 inches: the figure canvas
is 5 in wide and text sizes are the FINAL sizes (no rescaling).  Figure 1
uses a compact 7--8.2 pt hierarchy; Figure 2 retains its original hierarchy.
"""

import importlib.util
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from textwrap import fill  # noqa: E402

from dengyunetwork import plotstyle  # noqa: E402
from dengyunetwork.facts import build_target_case_control  # noqa: E402
from dengyunetwork.data import load_events, node_entry_times  # noqa: E402

EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures"
plotstyle.apply()
# final-size fonts: figures are embedded at 5 in width, so rcParams sizes
# are already the printed sizes
plotstyle.mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.facecolor": plotstyle.SURFACE,
})

FEATS = ["net_cap", "count_cap", "log_recency"]
FEAT_LABELS = {"net_cap": "net capital", "count_cap": "count capital",
               "log_recency": "log recency"}
DATASETS = ["OTC", "Alpha", "RfA"]
DCOLORS = {"OTC": plotstyle.C1, "Alpha": plotstyle.C2, "RfA": plotstyle.C3}
DMARKERS = {"OTC": "o", "Alpha": "s", "RfA": "^"}


def _rfa_module():
    """Load the RfA pipeline only when Figure 2 needs its strata."""
    spec = importlib.util.spec_from_file_location(
        "rfa", ROOT / "scripts" / "08_wiki_rfa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_otc_alpha():
    facts = json.loads((EXP_DIR / "facts.json").read_text())
    g1 = facts["target_selection_probe"]
    rob = json.loads((EXP_DIR / "wiki_rfa_robustness.json").read_text())
    alpha = rob["three_settings"]["alpha_g1"]
    rfa_r = rob["three_settings"]["rfa_aligned"]
    return {"OTC": g1, "Alpha": alpha, "RfA": rfa_r}


def feature_sds() -> dict:
    """Predictor standard deviations within each dataset's strata."""
    rfa = _rfa_module()
    sds = {}
    otc_df = load_events(str(ROOT / "data" / "bitcoin_otc" / "soc-sign-bitcoinotc.csv.gz"))
    alpha_df = load_events(str(ROOT / "data" / "bitcoin_alpha" / "soc-sign-bitcoinalpha.csv.gz"))
    for name, df in [("OTC", otc_df), ("Alpha", alpha_df)]:
        entry = node_entry_times(df)
        cc = build_target_case_control(df, entry, n_controls=30,
                                       max_events=12000,
                                       rng=np.random.default_rng(0))
        sds[name] = {f: float(cc[f].std()) for f in FEATS}
    ok = rfa.parse_rfa().dropna(subset=["src", "tgt", "timestamp"])
    ok = ok.sort_values("timestamp").reset_index(drop=True)
    ok["rating_n"] = ok["rating"].astype(float)
    t0, t1 = ok["timestamp"].min(), ok["timestamp"].max()
    cutoff = t0 + 0.7 * (t1 - t0)
    test = ok[ok["timestamp"] > cutoff]
    windows = rfa.build_windows(ok)
    nodes = np.unique(np.concatenate([ok["src"].values, ok["tgt"].values]))
    pos = {n: k for k, n in enumerate(nodes)}
    delta = rfa.LN2 / (180.0 * 86400.0)
    _, cc = rfa.window_target_choice(ok, test, windows, nodes, pos, delta,
                                     seed=0, return_strata=True,
                                     fit_model=False)
    sds["RfA"] = {f: float(cc[f].std()) for f in FEATS}
    return sds


def _box(ax, x, y, w, h, title, eq, keyword, title_size=7.9,
         eq_size=7.4, kw_size=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=plotstyle.SURFACE, ec=plotstyle.INK,
                                lw=1.0))
    ax.text(x + w / 2, y + h - 0.060, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold")
    ax.text(x + w / 2, y + 0.235, eq, ha="center", va="center",
            fontsize=eq_size)
    ax.text(x + w / 2, y + 0.060, keyword, ha="center", va="center",
            fontsize=kw_size, style="italic",
            color=plotstyle.INK_SECONDARY, linespacing=1.15)


def fig_theory_error(out: Path) -> None:
    # A little more height and an explicit horizontal safe area keep the
    # rounded borders visible after tight-bbox export and improve leading in
    # the error panel at the final 5-inch print width.
    W, H = 5.0, 2.40
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.grid(False)

    # ---- three process boxes ----
    x0, gap, y0, bh = 0.030, 0.065, 0.585, 0.380
    bw = (1 - 2 * x0 - 2 * gap) / 3
    _box(ax, x0, y0, bw, bh, "Microscopic events",
         r"$\lambda_{ij}^N=N^{-1}a_iW_{ij}\Psi_{ij}$",
         "signed directed events\none event per pair")
    _box(ax, x0 + (bw + gap), y0, bw, bh, r"$(C,H)$ state update",
         r"$C_j^+\!+\!=\!\eta R,\;\;H_j^+\!+\!=\;1$",
         r"decay $dC_j=-\delta C_j\,dt$" "\n"
         r"$dH_j=-\rho H_j\,dt$",
         eq_size=7.0)
    _box(ax, x0 + 2 * (bw + gap), y0, bw, bh, "Kinetic limit",
         r"$\partial_t f=\mathcal{D}(f)+\mathcal{J}(f)$",
         "$A_t$ sender field\ntype weights $\\pi$")
    for k in range(2):
        left_box_right = x0 + k * (bw + gap) + bw
        next_box_left = x0 + (k + 1) * (bw + gap)
        # A centred glyph is visually cleaner than an annotation over this
        # short distance and cannot intrude into either rounded border.
        ax.text((left_box_right + next_box_left) / 2, y0 + bh / 2,
                r"$\rightarrow$", ha="center", va="center",
                fontsize=8.3, color=plotstyle.INK)

    # ---- full-width error panel (v0.11: joint law carries log(1+N)/sqrt N,
    #      exclusion constant (eta + omega) Lambda^2) ----
    py, ph = 0.028, 0.495
    ax.add_patch(FancyBboxPatch((x0, py), 1 - 2 * x0, ph,
                                boxstyle="round,pad=0.008",
                                fc="#f6f5f0", ec=plotstyle.AXIS, lw=0.9))
    tx = x0 + 0.020
    ax.text(tx, py + ph - 0.052,
            r"Finite-time error ($B_i\equiv 1$: no entry/exit)",
            fontsize=8.2, fontweight="bold", va="top")
    # joint empirical measure: the critical-dimension rate of Theorem 3
    ax.text(tx, py + 0.320,
            r"$\sup_{0\leq t\leq T}\mathbb{E}\,W_1(\mu_t^N,f_t)\;\leq\;C_T\left\{"
            r"\frac{\log(1+N)}{\sqrt{N}}\;+\;\|W_N-W\|_\infty"
            r"\;\;+\;\;\varepsilon_{\mathrm{history}}\right\}$",
            fontsize=7.5, va="center")
    # Three columns make the decomposition legible at the final 5-inch width.
    cols = [
        (tx, r"$N^{-1/2}$", "particle coupling\n(Theorem 2)"),
        (0.355, r"$N^{-1/2}$", "capital and attention\nmarginals"),
        (0.690, r"$\varepsilon_{\mathrm{history}}$", "pair-history exclusion"),
    ]
    for xx, symbol, label in cols:
        ax.text(xx, py + 0.205, symbol, fontsize=7.5, fontweight="bold",
                va="center")
        ax.text(xx, py + 0.137, label, fontsize=7.0, va="center",
                color=plotstyle.INK_SECONDARY)
    ax.text(tx, py + 0.048,
            r"$\varepsilon_{\mathrm{history}}(N,T)\;\leq\;"
            r"(\eta+\omega)\,\Lambda^2T^2e^{L_XT}/(2N)=O(N^{-1})$"
            r"  ($\Lambda=AP_1\max\{1,P_T\}$; $T^2$: accumulated history)",
            fontsize=7.0, va="center", color=plotstyle.INK_SECONDARY)
    plotstyle.save(fig, str(out / "fig_theory_error.pdf"))
    plotstyle.save(fig, str(out / "fig_theory_error.png"))


def fig_three_settings(probes: dict, sds: dict, out: Path) -> None:
    # predictor-standardised coefficients: beta_std = beta * SD_x
    rows = []
    for feat in FEATS:
        for ds in DATASETS:
            p = probes[ds]
            b, se, sd = p["coef"][feat], p["se"][feat], sds[ds][feat]
            rows.append((feat, ds, b * sd, 1.96 * se * sd))

    W, H = 5.0, 3.12
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.20], wspace=0.19)
    ax = fig.add_subplot(gs[0])
    ax.grid(True, axis="y", color=plotstyle.GRIDLINE, lw=0.6)
    ax.grid(False, axis="x")
    ax.axvline(0.0, color=plotstyle.INK, lw=0.9)

    ypos = {}
    y = 0.0
    for feat in FEATS:
        for ds in DATASETS:
            ypos[(feat, ds)] = y
            y += 1.0
        y += 0.7
    n_rows = y - 0.7
    ax.set_ylim(-0.5, n_rows + 0.4)
    ax.set_xlim(-3.55, 0.75)
    # Put the feature names on the y-axis rather than in the data region.
    group_mid = [np.mean([ypos[(feat, ds)] for ds in DATASETS])
                 for feat in FEATS]
    ax.set_yticks(group_mid)
    ax.set_yticklabels([FEAT_LABELS[f] for f in FEATS], fontsize=8.0,
                       fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlabel("predictor-standardised coefficient "
                  r"($\beta\times\mathrm{SD}_x$)", fontsize=8.5)
    ax.set_title("Who receives the next rating/vote", fontsize=9.5,
                 loc="left", pad=6)

    for feat in FEATS:
        for ds in DATASETS:
            b, h = next((r[2], r[3]) for r in rows
                        if r[0] == feat and r[1] == ds)
            yy = ypos[(feat, ds)]
            ax.plot([b - h, b + h], [yy, yy], color=DCOLORS[ds], lw=1.4,
                    solid_capstyle="round")
            ax.plot([b], [yy], marker=DMARKERS[ds], ms=6.0,
                    color=DCOLORS[ds], mec="none")
    legend = [Line2D([], [], color=DCOLORS[ds], marker=DMARKERS[ds], ls="",
                     label=ds, ms=6) for ds in DATASETS]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, frameon=False, fontsize=8, handletextpad=0.4,
              columnspacing=0.9)

    # right panel: three cross-dataset conclusions only
    sax = fig.add_subplot(gs[1])
    sax.axis("off")
    sax.set_xlim(0, 1); sax.set_ylim(0, 1)
    items = [
        ("OTC -- Alpha (open markets)", "capital structure reproduces: "
         r"count $+0.21$ / $+0.22$, net $-0.10$ / $-0.08$"),
        ("RfA (institutional)", "capital signs reverse: "
         r"count $-0.19$, net $+0.13$"),
        ("All three datasets", r"recency dominates ($-2.0$, $-2.0$, $-2.5$)"),
    ]
    yy = 0.91
    for head, body in items:
        sax.text(0.02, yy, head, fontsize=8.3, fontweight="bold", va="top")
        yy -= 0.070
        sax.text(0.04, yy, fill(body, width=31), fontsize=7.7, va="top",
                 color=plotstyle.INK_SECONDARY)
        yy -= 0.205
    plotstyle.save(fig, str(out / "fig_three_settings.pdf"))
    plotstyle.save(fig, str(out / "fig_three_settings.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=("theory", "settings", "all"),
                        default="all")
    args = parser.parse_args()
    FIG_DIR.mkdir(exist_ok=True)
    if args.figure == "theory":
        fig_theory_error(FIG_DIR)
        return
    probes = load_otc_alpha()
    sds = feature_sds()
    if args.figure == "all":
        fig_theory_error(FIG_DIR)
    fig_three_settings(probes, sds, FIG_DIR)
    for feat in FEATS:
        for ds in DATASETS:
            p = probes[ds]
            b, se, sd = p["coef"][feat], p["se"][feat], sds[ds][feat]
            print(f"{feat:12s} {ds:6s} b*SD={b*sd:+.4f} "
                  f"CI={1.96*se*sd:.4f}")


if __name__ == "__main__":
    main()
