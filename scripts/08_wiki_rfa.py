#!/usr/bin/env python
"""Wiki-RfA cross-context robustness validation (third dataset).

Per the project's validation rules (gates_log.md G5 follow-up):
  1. Wiki-RfA is fitted SEPARATELY and never pooled with Bitcoin OTC/Alpha.
  2. The capital construct differs: institutional reputation / legitimacy
     capital under an open adminship window, not peer trust ratings.
  3. The once-per-pair exclusion is OFF (repeated voter-candidate pairs are
     legal; this corresponds to the no-exclusion case of the main theorem,
     epsilon_history = 0).
  4. The risk set is restricted to candidates with an open RfA window,
     reconstructed from each candidate's vote time series.  If the windows
     cannot be reconstructed defensibly, the task degrades to vote
     sign/arrival and capital-distribution dynamics only.
  5. Election outcomes (RES) provide an additional macro-level validation.

Metrics:
  - sign model: support/neutral/oppose prediction from capital covariates
  - arrival-rate trajectory: kinetic solver vs realised (per-capita/day)
  - capital distribution evolution: W1/JSD of the event-weighted capital
  - election outcome: does the candidate's end-of-window capital predict RES?
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dengyunetwork.facts import ETA, CapitalEngine, LN2  # noqa: E402

EXP_DIR = ROOT / "experiments"
DATA = ROOT / "data" / "wiki_rfa" / "wiki-RfA.txt.gz"

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
# datetime months are 1-based: index + 1
MONTHS = {m.lower(): i + 1 for i, m in enumerate(_MONTHS)}
# both full and abbreviated names
MONTHS.update({m.lower()[:3]: i + 1 for i, m in enumerate(_MONTHS)})


def parse_timestamp(s: str) -> float | None:
    """'23:13, 19 April 2013' -> unix seconds (UTC assumed)."""
    m = re.match(r"(\d{1,2}):(\d{2}),\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s.strip())
    if not m:
        return None
    hh, mm, dd, mon, yy = m.groups()
    if int(hh) > 23 or int(mm) > 59 or mon.lower() not in MONTHS:
        return None
    try:
        dt = datetime(int(yy), MONTHS[mon.lower()], int(dd), int(hh), int(mm),
                      tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def parse_rfa(path: Path = DATA) -> pd.DataFrame:
    """State-machine parser: SRC opens a record, TGT/VOT/RES/YEA/DAT fill it,
    TXT is skipped (multi-line until the next SRC)."""
    records = []
    cur = {}
    with __import__("gzip").open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("SRC:"):
                if cur.get("src"):
                    records.append(cur)
                cur = {"src": line[4:].strip()}
            elif line.startswith("TGT:"):
                cur["tgt"] = line[4:].strip()
            elif line.startswith("VOT:"):
                cur["vot"] = line[4:].strip()
            elif line.startswith("RES:"):
                cur["res"] = line[4:].strip()
            elif line.startswith("DAT:"):
                cur["dat"] = line[4:].strip()
            elif line.startswith("YEA:"):
                cur["yea"] = line[4:].strip()
    if cur.get("src"):
        records.append(cur)
    df = pd.DataFrame(records)
    df["rating"] = pd.to_numeric(df["vot"], errors="coerce").fillna(0).astype(int)
    df["result"] = pd.to_numeric(df["res"], errors="coerce")
    df["timestamp"] = df["dat"].map(parse_timestamp)
    df["year"] = pd.to_numeric(df["yea"], errors="coerce")
    return df


def audit(df: pd.DataFrame) -> dict:
    ok = df.dropna(subset=["src", "tgt", "timestamp"])
    pairs = ok["src"] + "->" + ok["tgt"]
    return {
        "raw_records": int(len(df)),
        "complete": int(len(ok)),
        "users": int(pd.unique(pd.concat([ok["src"], ok["tgt"]])).size),
        "targets": int(ok["tgt"].nunique()),
        "distinct_pairs": int(pairs.nunique()),
        "repeated_pairs": int(pairs.value_counts()[pairs.value_counts() > 1].size),
        "events_after_first": int(len(ok) - pairs.nunique()),
        "support": int((ok["rating"] == 1).sum()),
        "neutral": int((ok["rating"] == 0).sum()),
        "oppose": int((ok["rating"] == -1).sum()),
        "t_min": float(ok["timestamp"].min()),
        "t_max": float(ok["timestamp"].max()),
        "unparseable_dates": int(df["timestamp"].isna().sum()),
    }


def build_windows(df: pd.DataFrame, gap_days: float = 60.0) -> pd.DataFrame:
    """Reconstruct RfA windows per candidate: cluster the candidate's vote
    times into rounds separated by > gap_days; each round is one window
    [t_first - buffer, t_last + buffer].  Returns one row per (candidate,
    round) with window bounds and the outcome RES of the round (majority
    vote of the round)."""
    df = df.sort_values("timestamp")
    rows = []
    for tgt, g in df.groupby("tgt"):
        t = g["timestamp"].values
        res = g["result"].values
        start = 0
        for k in range(1, len(t) + 1):
            if k == len(t) or (t[k] - t[k - 1]) > gap_days * 86400.0:
                ts, te = t[start], t[k - 1]
                r = res[start:k]
                r = r[~np.isnan(r)]
                outcome = int(np.median(r)) if len(r) else np.nan
                rows.append({"tgt": tgt, "t_start": ts, "t_end": te,
                             "n_votes": k - start, "outcome": outcome})
                start = k
    return pd.DataFrame(rows)


def main() -> None:
    t_start = time.time()
    EXP_DIR.mkdir(exist_ok=True)
    df = parse_rfa()
    out: dict = {"audit": audit(df)}
    print(json.dumps(out["audit"], indent=1))

    ok = df.dropna(subset=["src", "tgt", "timestamp"]).copy()
    ok = ok.sort_values("timestamp").reset_index(drop=True)
    # normalise capital jumps: r in {-1, 0, +1} (neutral = 0)
    ok["rating_n"] = ok["rating"].astype(float)

    # ---- windows ---------------------------------------------------------
    windows = build_windows(ok)
    out["windows"] = {
        "n_windows": int(len(windows)),
        "n_candidates_with_windows": int(windows["tgt"].nunique()),
        "median_votes_per_window": float(windows["n_votes"].median()),
        "median_window_days": float(((windows["t_end"] - windows["t_start"])
                                     / 86400.0).median()),
        "outcome_1_share": float((windows["outcome"] == 1).mean()),
    }
    print("windows:", json.dumps(out["windows"]))
    win_end = dict(zip(windows["tgt"], windows["t_end"]))  # per (tgt, latest)
    win_start = dict(zip(windows["tgt"], windows["t_start"]))

    # ---- split (70/30 of the time span) ----------------------------------
    t0, t1 = ok["timestamp"].min(), ok["timestamp"].max()
    cutoff = t0 + 0.7 * (t1 - t0)
    train = ok[ok["timestamp"] <= cutoff]
    test = ok[ok["timestamp"] > cutoff]
    out["split"] = {"cutoff": float(cutoff), "train": int(len(train)),
                    "test": int(len(test))}
    print("split:", json.dumps(out["split"]))
    delta = LN2 / (180.0 * 86400.0)

    # ---- capital engine over the full stream (no exclusion) --------------
    nodes = np.unique(np.concatenate([ok["src"].values, ok["tgt"].values]))
    pos = {n: k for k, n in enumerate(nodes)}
    N = len(nodes)
    eng = CapitalEngine(N, delta)
    last_t = np.full(N, -np.inf)
    caps = np.zeros(len(ok))
    for e, row in enumerate(ok.itertuples(index=False)):
        j = pos[row.tgt]
        if row.timestamp > last_t[j]:
            eng.value[j] *= np.exp(-delta * (row.timestamp - last_t[j]))
            last_t[j] = row.timestamp
        caps[e] = eng.value[j]
        eng.value[j] += ETA * row.rating_n
    ok["net_cap"] = caps

    # ---- sign model (support / neutral / oppose) -------------------------
    from sklearn.linear_model import LogisticRegression

    tr = ok.iloc[: int(0.7 * len(ok))]
    te = ok.iloc[int(0.7 * len(ok)):]
    X_tr = tr[["net_cap"]].values
    y_tr = (tr["rating_n"].values > 0).astype(int)
    X_te = te[["net_cap"]].values
    y_te = (te["rating_n"].values > 0).astype(int)
    lr = LogisticRegression(C=1e6, max_iter=3000).fit(X_tr, y_tr)
    p_te = lr.predict_proba(X_te)[:, 1]
    ll = float(np.mean(y_te * np.log(np.clip(p_te, 1e-9, 1))
                       + (1 - y_te) * np.log(np.clip(1 - p_te, 1e-9, 1))))
    base_rate = float(y_te.mean())
    ll0 = float(base_rate * np.log(base_rate)
                + (1 - base_rate) * np.log(1 - base_rate))
    # three-class with neutral as its own class
    lr3 = LogisticRegression(C=1e6, max_iter=3000).fit(
        np.column_stack([tr["net_cap"].values,
                         tr["timestamp"].values / 86400 / 365]),
        tr["rating_n"].values.astype(int) + 1)
    acc3 = float((lr3.predict(np.column_stack(
        [te["net_cap"].values, te["timestamp"].values / 86400 / 365]))
        == te["rating_n"].values.astype(int) + 1).mean())
    out["sign_model"] = {
        "coef_net_cap": float(lr.coef_[0][0]),
        "test_logloss_vs_baseline": {"model": round(-ll, 4),
                                     "baseline": round(-ll0, 4)},
        "three_class_accuracy": round(acc3, 4),
        "base_support_rate": round(base_rate, 4),
    }
    print("sign model:", json.dumps(out["sign_model"]))

    # ---- window-restricted target choice (the OTC core metric) -----------
    out["window_target_choice"] = window_target_choice(
        ok, test, windows, nodes, pos, delta)

    # ---- arrival-rate trajectory (capital-intensity model, no exclusion) -
    out["trajectory"] = arrival_trajectory(ok, train, test, nodes, pos, delta,
                                           cutoff, ETA)

    # ---- election outcome validation -------------------------------------
    out["election"] = election_validation(ok, windows, delta)

    out["wall_seconds"] = round(time.time() - t_start, 1)
    (EXP_DIR / "wiki_rfa.json").write_text(json.dumps(out, indent=2))
    print(f"\n[ok] wrote experiments/wiki_rfa.json ({out['wall_seconds']}s)")


def window_target_choice(ok, test, windows, nodes, pos, delta,
                         n_controls: int = 30, max_events: int = 12000,
                         seed: int = 0,
                         recency_cap_days: float | None = None,
                         min_window_votes: int = 1,
                         return_strata: bool = False,
                         fit_model: bool = True):
    """Window-restricted target choice: the risk set at time t is the set of
    candidates with an OPEN RfA window (any round; rounds are reconstructed
    per candidate, so a candidate with several candidacy rounds enters the
    risk set during each of them -- the v0.4 bug of keeping only the LAST
    round per candidate is fixed).  Case-control conditional logit on the
    OTC-aligned feature set (net capital, count capital, log indegree, age,
    log recency, ever sent) answers: within the candidate risk set, does
    capital predict who receives the vote?  This is the direct cross-context
    counterpart of the OTC target-choice probe (G1).  Sender-activity is
    constant within a stratum (one voter) and is therefore not identifiable
    in the conditional logit, as in the OTC case-control probe."""
    import heapq
    rng = np.random.default_rng(seed)
    N = len(nodes)
    # per-candidate window list (ALL rounds), sorted by t_start;
    # robustness variants: drop windows with few votes (min_window_votes)
    per_cand: dict[int, list] = {}
    for w in windows.itertuples(index=False):
        if w.n_votes < min_window_votes:
            continue
        per_cand.setdefault(pos[w.tgt], []).append((w.t_start, w.t_end))
    for lst in per_cand.values():
        lst.sort()
    starts = sorted((ts, te, k) for k, lst in per_cand.items()
                    for (ts, te) in lst)
    si = 0
    open_heap: list = []  # (t_end, candidate) min-heap of open windows

    eng = CapitalEngine(N, delta)
    eng_cnt = CapitalEngine(N, delta)
    indeg = np.zeros(N, dtype=int)
    sent = np.zeros(N, dtype=int)
    first_seen = np.full(N, np.inf)
    last_received = np.full(N, -np.inf)
    last_t = np.full(N, -np.inf)
    voted_round: dict[int, dict[int, float]] = {}  # voter -> {target: round_end}
    rows = []
    test_ts = set(test["timestamp"].values)
    n_strata = 0
    for row in ok.itertuples(index=False):
        i, j, t = pos[row.src], pos[row.tgt], row.timestamp
        if first_seen[j] == np.inf:
            first_seen[j] = t
        # ---- maintain the set of candidates with an open window at t ----
        while si < len(starts) and starts[si][0] <= t:
            heapq.heappush(open_heap, (starts[si][1], starts[si][2]))
            si += 1
        while open_heap and open_heap[0][0] < t:
            heapq.heappop(open_heap)
        if t in test_ts and len(open_heap) > 5:
            # risk set: candidates with an open round; exclude rounds the
            # voter already voted in (still open); j is always kept
            vd = voted_round.get(i, {})
            others = [c for (te, c) in open_heap
                      if c != j and not (vd.get(c, -np.inf) >= t)]
            if len(others) >= 3:
                ctrl = rng.choice(np.asarray(others, dtype=int),
                                  size=min(n_controls, len(others)),
                                  replace=False)
                cand = np.concatenate([[j], ctrl])
                case = np.zeros(len(cand)); case[0] = 1.0
                net_at = np.array([eng.at(c, t) for c in cand])
                cnt_at = np.array([eng_cnt.at(c, t) for c in cand])
                age = np.array([(t - first_seen[c]) / 86400.0
                                if np.isfinite(first_seen[c]) else 0.0
                                for c in cand])
                rec_raw = np.array([(t - last_received[c]) / 86400.0
                                    if np.isfinite(last_received[c]) else 0.0
                                    for c in cand])
                if recency_cap_days is not None:
                    rec_raw = np.minimum(rec_raw, recency_cap_days)
                rec = np.where(np.isfinite(rec_raw), rec_raw, age)
                rows.append(pd.DataFrame({
                    "stratum": n_strata, "case": case, "net_cap": net_at,
                    "count_cap": cnt_at,
                    "log_indegree": np.log1p(indeg[cand]),
                    "age_days": age,
                    "log_recency": np.log1p(np.maximum(rec, 0.0)),
                    "ever_sent": (sent[cand] > 0).astype(float),
                }))
                n_strata += 1
                if n_strata >= max_events:
                    break
        # ---- state update with the realised vote (no exclusion: repeated
        # voter-candidate pairs are a new candidacy round) ----
        if t > last_t[j]:
            eng.value[j] *= np.exp(-delta * (t - last_t[j]))
            last_t[j] = t
        eng.value[j] += ETA * row.rating_n
        eng_cnt.value[j] += 1.0
        indeg[j] += 1
        sent[i] += 1
        last_received[j] = t
        # round bookkeeping: remember the end of j's current open round
        vd = voted_round.setdefault(i, {})
        te_j = next((te for (ts, te) in per_cand.get(j, ()) if ts <= t <= te),
                    np.inf)
        vd[j] = te_j
    if not rows:
        return {"n_strata": 0, "note": "no usable strata"}
    cc = pd.concat(rows, ignore_index=True)
    if not fit_model:
        # Figure construction needs only the predictor dispersion in the
        # reconstructed strata.  Avoid an unnecessary second optimisation,
        # which can emit numerical overflow warnings on this large design.
        return ({"n_strata": n_strata}, cc) if return_strata else {
            "n_strata": n_strata
        }

    from statsmodels.discrete.conditional_models import ConditionalLogit

    feats = ["net_cap", "count_cap", "log_indegree", "age_days",
             "log_recency", "ever_sent"]
    m = ConditionalLogit(endog=cc["case"], exog=cc[feats], groups=cc["stratum"])
    res = m.fit(disp=False, maxiter=300)
    conv = getattr(res, "mle_retvals", None)
    converged = bool(conv.get("converged", True)) if isinstance(conv, dict) \
        else True
    out = {
        "n_strata": n_strata,
        "converged": converged,
        "coef": {f: float(res.params[f]) for f in feats},
        "se": {f: float(res.bse[f]) for f in feats},
        "pvalue": {f: float(res.pvalues[f]) for f in feats},
    }
    if return_strata:
        return out, cc
    return out


def arrival_trajectory(ok, train, test, nodes, pos, delta, cutoff, eta):
    """Per-capita daily vote rate: realised vs a capital-driven intensity
    model (activity x kernel-free target choice), binned over the test."""
    N = len(nodes)
    t0_test = test["timestamp"].min()
    t1_test = test["timestamp"].max()
    T_days = (t1_test - t0_test) / 86400.0
    n_bins = 16
    bin_edges = np.linspace(0, T_days, n_bins + 1)
    h_real, _ = np.histogram((test["timestamp"].values - t0_test) / 86400.0,
                             bins=bin_edges)
    real_rates = h_real / (T_days / n_bins) / N
    # simple arrival model: per-capita rate proportional to active capital
    # mass (rich-get-richer free test of the capital channel)
    eng2 = CapitalEngine(N, delta)
    last2 = np.full(N, -np.inf)
    pred = np.zeros(n_bins)
    # capital state at test start
    for row in train.itertuples(index=False):
        j = pos[row.tgt]
        if row.timestamp > last2[j]:
            eng2.value[j] *= np.exp(-delta * (row.timestamp - last2[j]))
            last2[j] = row.timestamp
        eng2.value[j] += eta * row.rating_n
    # evolve through the test events; model rate = capital intensity:
    # lambda(t) ~ (1/N) sum_i psi2(C_i(t)) with psi2 = 1 + tanh (bounded)
    ev = test.itertuples(index=False)
    next_ev = next(ev, None)
    psi2 = lambda c: 1.0 + np.tanh(0.5 * np.asarray(c, dtype=float))
    for b in range(n_bins):
        t_e = t0_test + (b + 1) * (t1_test - t0_test) / n_bins
        acc = 0.0
        cnt = 0
        while next_ev is not None and next_ev.timestamp <= t_e:
            j = pos[next_ev.tgt]
            if next_ev.timestamp > last2[j]:
                eng2.value[j] *= np.exp(-delta * (next_ev.timestamp - last2[j]))
                last2[j] = next_ev.timestamp
            acc += psi2(eng2.value).sum()
            cnt += 1
            eng2.value[j] += eta * next_ev.rating_n
            next_ev = next(ev, None)
        pred[b] = acc / max(cnt, 1) / N if cnt else 0.0   # per-capita
    # scale-match the level (a single constant), compare the SHAPE
    s = float(np.sum(real_rates * pred) / np.sum(pred * pred)) if pred.sum() else 0
    shape_diff = float(np.mean(np.abs(real_rates - s * pred)
                               / np.maximum(real_rates, 1e-9)))
    corr_shape = float(np.corrcoef(real_rates, pred)[0, 1]) if len(pred) > 1 else 0
    return {"real_rates": [round(x, 6) for x in real_rates],
            "pred_rates": [round(x, 6) for x in pred],
            "shape_scale_s": round(s, 4),
            "rel_mean_abs_diff_after_scale": round(shape_diff, 4),
            "shape_corr": round(corr_shape, 4)}


def election_validation(ok, windows, delta):
    """Does the candidate's PRE-WINDOW capital predict the outcome?

    Clean reconstruction (audit fix): the start capital of each (candidate,
    round) is recorded at the FIRST in-window event, BEFORE that event's
    vote is applied -- so it contains only votes from previous rounds and
    is never contaminated by the current round.  Each round of a candidate
    is recorded separately (the v0.3 code keyed by candidate only, which
    collapsed multiple rounds into one, and applied the first vote before
    recording the "pre-window" capital).  We also record the end-of-window
    capital (which does contain the current round's votes) for contrast."""
    from sklearn.metrics import roc_auc_score

    nodes = np.unique(np.concatenate([ok["src"].values, ok["tgt"].values]))
    pos = {n: k for k, n in enumerate(nodes)}
    eng = CapitalEngine(len(nodes), delta)
    last_t = np.full(len(nodes), -np.inf)
    # per-candidate sorted windows (ALL rounds)
    queries: dict[int, list] = {}
    for w in windows.itertuples(index=False):
        queries.setdefault(pos[w.tgt], []).append((w.t_start, w.t_end, w.outcome))
    for q in queries.values():
        q.sort()
    cap_start: dict[tuple, float] = {}
    cap_end: dict[tuple, float] = {}
    oc_of: dict[tuple, float] = {}
    for row in ok.itertuples(index=False):
        j = pos[row.tgt]
        t = row.timestamp
        if t > last_t[j]:
            eng.value[j] *= np.exp(-delta * (t - last_t[j]))
            last_t[j] = t
        qs = queries.get(j, ())
        # pre-window capital: record BEFORE applying this event, for rounds
        # whose first in-window event is this one (t == ts, since windows
        # start at their first vote); value excludes all round-1 votes.
        # The bound is t <= te so that zero-length windows (a single vote,
        # ts == te) are recorded too.
        for wi, (ts, te, oc) in enumerate(qs):
            if (j, wi) not in cap_start and ts <= t <= te:
                cap_start[(j, wi)] = eng.value[j]
                oc_of[(j, wi)] = oc
            # end-of-window capital for rounds already finished: the value
            # contains the full round's votes decayed to t; decay back to te
            # (no events of j between te and t, as this is the first event
            # after the round) -- recorded before adding this event
            if (j, wi) not in cap_end and te < t:
                cap_end[(j, wi)] = eng.value[j] * np.exp(-delta * (t - te))
        eng.value[j] += ETA * row.rating_n
        # end-of-window capital exactly at the last in-window vote (t == te)
        for wi, (ts, te, oc) in enumerate(qs):
            if (j, wi) not in cap_end and t == te:
                cap_end[(j, wi)] = eng.value[j]
    d = pd.DataFrame([{"cap": v, "outcome": oc_of[k]}
                      for k, v in cap_start.items()])
    d = d[d["outcome"].isin([-1, 1])]
    if len(d) < 50:
        return {"n_windows": int(len(d)), "note": "too few windows"}
    auc_all = roc_auc_score((d["outcome"] == 1).astype(int), d["cap"])
    share_zero = float((d["cap"] == 0).mean())
    d_nz = d[d["cap"] != 0]
    auc_nz = None
    if len(d_nz) >= 20 and d_nz["outcome"].nunique() == 2:
        auc_nz = roc_auc_score((d_nz["outcome"] == 1).astype(int),
                               d_nz["cap"])
    return {
        "n_windows": int(len(d)),
        "share_zero_pre_window_capital": round(share_zero, 4),
        "auc_pre_window_capital": round(auc_all, 4),
        "n_nonzero_pre_window": int(len(d_nz)),
        "auc_pre_window_capital_nonzero": round(auc_nz, 4)
        if auc_nz is not None else None,
        "note": "pre-window capital excludes the current round's votes "
                "(clean predictor; recorded before the round's first vote); "
                "the previous AUC 0.796/0.878 were contaminated by the first "
                "vote of the round and are retracted",
        "success_rate": float((d["outcome"] == 1).mean()),
    }


if __name__ == "__main__":
    main()
