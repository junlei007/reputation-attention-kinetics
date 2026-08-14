#!/usr/bin/env python
"""Two main-text figures.

fig_theory_error:  micro -- capital -- kinetic -- finite-time error, with
  the three-way decomposition N^{-1/2} + ||W_N-W||_inf + eps_history
  (eps_history = O(N^{-1})) on the right.  Answers "what is the theory
  contribution".

fig_three_settings:  fully standardised conditional-logit coefficients
  (beta * SD of the feature in the strata data) with 95% CI for the three
  features net capital / count capital / log recency, across the three
  datasets OTC, Alpha, RfA, plus a summary panel.  Answers "what did the
  empirical boundary actually find".

Both figures use the project palette (plotstyle.py); light mode.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from dengyunetwork import plotstyle  # noqa: E402
from dengyunetwork.facts import build_target_case_control  # noqa: E402
from dengyunetwork.data import load_events, node_entry_times  # noqa: E402

spec = importlib.util.spec_from_file_location("rfa", ROOT / "scripts" / "08_wiki_rfa.py")
rfa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rfa)

EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures"
plotstyle.apply()

FEATS = ["net_cap", "count_cap", "log_recency"]
FEAT_LABELS = {"net_cap": "net capital", "count_cap": "count capital",
               "log_recency": "log recency"}
DATASETS = ["OTC", "Alpha", "RfA"]
DCOLORS = {"OTC": plotstyle.C1, "Alpha": plotstyle.C2, "RfA": plotstyle.C3}


def load_otc_alpha():
    facts = json.loads((EXP_DIR / "facts.json").read_text())
    g1 = facts["target_selection_probe"]
    rob = json.loads((EXP_DIR / "wiki_rfa_robustness.json").read_text())
    alpha = rob["three_settings"]["alpha_g1"]
    rfa_r = rob["three_settings"]["rfa_aligned"]
    return {"OTC": g1, "Alpha": alpha, "RfA": rfa_r}


def feature_sds() -> dict:
    """Feature standard deviations in the strata data of each dataset."""
    sds = {}
    # OTC / Alpha: same case-control design as the G1 probe
    otc_df = load_events(str(ROOT / "data" / "bitcoin_otc" / "soc-sign-bitcoinotc.csv.gz"))
    alpha_df = load_events(str(ROOT / "data" / "bitcoin_alpha" / "soc-sign-bitcoinalpha.csv.gz"))
    for name, df in [("OTC", otc_df), ("Alpha", alpha_df)]:
        entry = node_entry_times(df)
        cc = build_target_case_control(df, entry, n_controls=30,
                                       max_events=12000,
                                       rng=np.random.default_rng(0))
        sds[name] = {f: float(cc[f].std()) for f in FEATS}
    # RfA: aligned probe, full feature set
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
                                     seed=0, return_strata=True)
    sds["RfA"] = {f: float(cc[f].std()) for f in FEATS}
    return sds


def fig_theory_error(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.grid(False)

    boxes = [
        (0.02, 0.40, 0.21, 0.52, "Microscopic marked events",
         r"$\lambda_{ij}^N=\frac{1}{N}\,B_iB_j(1-A_{ij})\,a_iW_{ij}$"
         r"$\;\psi_1(C_i)\psi_2(C_j)$",
         r"signed rating of ordered pair $(i,j)$; once-per-pair exclusion"),
        (0.27, 0.40, 0.21, 0.52, "Capital update",
         r"$C_j^+\leftarrow C_j+\eta R$,\; $R\sim Q$",
         r"decay $dC_j=-\delta C_j\,dt$; bounded marks $|R|\leq 1$"),
        (0.52, 0.40, 0.21, 0.52, "Kinetic master equation",
         r"$\partial_t f_{k,t}=\delta\partial_c(cf_{k,t})-\kappa_t f_{k,t}$"
         r"$+\;$ jump gain",
         r"$\kappa_t(k,c)=A_t(k)\psi_2(c)$, sender field $A_t$; "
         r"type proportions $\pi$"),
        (0.77, 0.40, 0.21, 0.52, "Finite-time error",
         r"$\sup_{t\leq T}\mathbb{E}\,W_1(\mu_t^N,f_t)\leq C_T\{\cdots\}$",
         r"three-way decomposition on the right"),
    ]
    for x, y, w, h, title, eq, note in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.012",
                                    fc=plotstyle.SURFACE, ec=plotstyle.INK,
                                    lw=1.1))
        ax.text(x + w / 2, y + h - 0.10, title, ha="center", va="top",
                fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, y + 0.24, eq, ha="center", va="center",
                fontsize=9.2)
        ax.text(x + w / 2, y + 0.075, note, ha="center", va="center",
                fontsize=7.2, color=plotstyle.INK_SECONDARY)
    for xa, xb in [(0.235, 0.268), (0.485, 0.518), (0.735, 0.768)]:
        ax.annotate("", xy=(xb, 0.66), xytext=(xa, 0.66),
                    arrowprops=dict(arrowstyle="-|>", color=plotstyle.INK,
                                    lw=1.4))

    # error decomposition panel on the right column
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.32,
                                boxstyle="round,pad=0.01",
                                fc="#f6f5f0", ec=plotstyle.AXIS, lw=0.9))
    ax.text(0.05, 0.285,
            r"$\sup_{t\leq T}\mathbb{E}\,W_1(\mu_t^N,f_t)\;\leq\;C_T\left\{"
            r"N^{-1/2}\;+\;\|W_N-W\|_\infty\;+\;\varepsilon_{\mathrm{history}}"
            r"\right\}$",
            fontsize=11.5, va="center")
    ax.text(0.05, 0.215, r"$N^{-1/2}$" + "  --  particle fluctuation",
            fontsize=9.8, va="center")
    ax.text(0.05, 0.165,
            r"$\|W_N-W\|_\infty$" + "  --  kernel approximation",
            fontsize=9.8, va="center")
    ax.text(0.05, 0.115,
            r"$\varepsilon_{\mathrm{history}}$" +
            "  --  once-per-pair exclusion",
            fontsize=9.8, va="center")
    ax.text(0.05, 0.055,
            r"$\varepsilon_{\mathrm{history}}(N,T)\;\leq\;"
            r"\eta\Lambda^2T^2e^{LT}/(2N)\;=\;O(N^{-1})$: "
            r"pair-specific memory beyond fixed finite-state graphon "
            r"bounds (Allmeier--Gast 2025)",
            fontsize=8.2, va="center", color=plotstyle.INK_SECONDARY)
    plotstyle.save(fig, str(out / "fig_theory_error.pdf"))
    plotstyle.save(fig, str(out / "fig_theory_error.png"))


def fig_three_settings(probes: dict, sds: dict, out: Path) -> None:
    # fully standardised coefficients: beta_std = beta * SD(feature)
    rows = []  # (feature, dataset, beta_std, ci_half)
    for feat in FEATS:
        for ds in DATASETS:
            p = probes[ds]
            b = p["coef"][feat]
            se = p["se"][feat]
            sd = sds[ds][feat]
            rows.append((feat, ds, b * sd, 1.96 * se * sd))

    fig = plt.figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax.grid(True, axis="y", color=plotstyle.GRIDLINE, lw=0.6)
    ax.grid(False, axis="x")
    ax.axvline(0.0, color=plotstyle.INK, lw=1.0)

    ypos = {}
    y = 0.0
    for feat in FEATS:
        for ds in DATASETS:
            ypos[(feat, ds)] = y
            y += 1.0
        y += 0.7  # feature-group gap
    n_rows = y - 0.7
    ax.set_ylim(-0.5, n_rows + 0.4)
    ax.set_xlim(-3.6, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("fully standardised conditional-logit coefficient "
                  r"($\beta\times\mathrm{SD}$)", fontsize=10)
    ax.set_title("Who receives the next rating/vote: capital and recency",
                 fontsize=11, loc="left", pad=8)

    for feat in FEATS:
        for ds in DATASETS:
            f_, ds_, b, h = next(r for r in rows
                                 if r[0] == feat and r[1] == ds)
            yy = ypos[(feat, ds)]
            ax.plot([b - h, b + h], [yy, yy], color=DCOLORS[ds], lw=1.6,
                    solid_capstyle="round")
            ax.plot([b], [yy], marker="o", ms=7.5, color=DCOLORS[ds],
                    mec="none")
    # feature group labels in the right margin of the data area
    for i, feat in enumerate(FEATS):
        ymid = ypos[(feat, "RfA")] + 0.5
        ax.text(1.35, ymid, FEAT_LABELS[feat], fontsize=9.5, ha="left",
                va="center", fontweight="bold", color=plotstyle.INK)
    ax.text(1.35, n_rows + 0.15, "feature", fontsize=8, ha="left",
            va="center", color=plotstyle.MUTED)

    legend = [Line2D([], [], color=DCOLORS[ds], marker="o", ls="",
                     label=ds, ms=7) for ds in DATASETS]
    ax.legend(handles=legend, loc="lower right", frameon=False)

    # summary panel
    sax = fig.add_subplot(gs[1])
    sax.axis("off")
    sax.set_xlim(0, 1); sax.set_ylim(0, 1)
    lines = [
        ("Open markets (OTC, Alpha)", "capital structure reproduces: "
         "count $+0.033$ / $+0.034$, net $-0.064$ / $-0.053$", True),
        ("Institutional (RfA)", "capital signs reverse: count $-0.004$, "
         "net $+0.032$", True),
        ("All three datasets", "recency is the dominant predictor "
         "($-1.2$, $-1.2$, $-2.5$)", True),
        ("RfA, capital as direction", "sign log loss $0.604\\to0.552$ "
         "($-8.6\\%$ vs baseline)", False),
        ("RfA, capital as selection", "pre-window AUC $0.534$; "
         "$88\\%$ of candidacies have zero pre-window capital", False),
        ("Arrival-rate shape", "fails in both contexts (recency/window "
         "driven)", False),
    ]
    yy = 0.97
    for label, body, is_head in lines:
        sax.text(0.02, yy, label, fontsize=9.3, fontweight="bold",
                 va="top", color=plotstyle.INK if is_head else plotstyle.C3)
        yy -= 0.055
        sax.text(0.06, yy, body, fontsize=8.2, va="top",
                 color=plotstyle.INK_SECONDARY)
        yy -= 0.085 if is_head else 0.075
    sax.text(0.02, yy - 0.03,
             "Target selection is governed by recency in all three "
             "datasets; accumulated capital retains limited predictive "
             "value only as a directional signal.",
             fontsize=8.8, va="top", style="italic",
             color=plotstyle.INK)
    plotstyle.save(fig, str(out / "fig_three_settings.pdf"))
    plotstyle.save(fig, str(out / "fig_three_settings.png"))


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    probes = load_otc_alpha()
    sds = feature_sds()
    fig_theory_error(FIG_DIR)
    fig_three_settings(probes, sds, FIG_DIR)
    # echo the standardised coefficients for the record
    for feat in FEATS:
        for ds in DATASETS:
            p = probes[ds]
            b, se, sd = p["coef"][feat], p["se"][feat], sds[ds][feat]
            print(f"{feat:12s} {ds:6s} beta*SD={b*sd:+.4f} "
                  f"CI_half={1.96*se*sd:.4f} (raw beta={b:+.4f}, SD={sd:.3f})")


if __name__ == "__main__":
    main()
