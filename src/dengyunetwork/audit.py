"""Data integrity audit reproducing the numbers in data/bitcoin_otc/README.md."""

from __future__ import annotations

import json

import numpy as np

from .data import load_events, sort_by_time


def audit(df=None) -> dict:
    """Audit the raw event table; returns a plain dict of verified numbers."""
    df = sort_by_time(load_events()) if df is None else sort_by_time(df)

    n_rows = len(df)
    n_users = len(np.unique(np.concatenate([df["source"].values, df["target"].values])))
    n_positive = int((df["rating"] > 0).sum())
    n_negative = int((df["rating"] < 0).sum())
    n_zero = int((df["rating"] == 0).sum())

    # malformed rows: any NaN or wrong dtypes
    malformed = int(df.isna().any(axis=1).sum())
    malformed += int((df["rating"].abs() > 10).sum())

    # self-loops
    self_loops = int((df["source"] == df["target"]).sum())

    # unique directed pairs: if n_rows == n_unique_pairs, no directed pair repeats
    pair_keys = df["source"].values.astype(np.int64) * 10_000_000 + df["target"].values
    n_unique_pairs = len(np.unique(pair_keys))

    # time range (timestamps are float64 seconds in this release)
    t0, t1 = float(df["timestamp"].min()), float(df["timestamp"].max())
    dt = np.diff(np.sort(df["timestamp"].values))
    min_gap_s = float(dt.min()) if len(dt) else None
    max_gap_s = float(dt.max()) if len(dt) else None
    n_ties = int((dt == 0).sum())

    result = {
        "n_rows": n_rows,
        "n_users": n_users,
        "n_positive_ratings": n_positive,
        "n_negative_ratings": n_negative,
        "n_zero_ratings": n_zero,
        "n_malformed": malformed,
        "n_self_loops": self_loops,
        "n_unique_directed_pairs": n_unique_pairs,
        "no_repeated_directed_pair": bool(n_rows == n_unique_pairs),
        "t_min_unix": t0,
        "t_max_unix": t1,
        "span_days": round((t1 - t0) / 86400.0, 1),
        "min_interevent_gap_s": min_gap_s,
        "max_interevent_gap_s": max_gap_s,
        "n_timestamp_ties": n_ties,
        "rating_min": int(df["rating"].min()),
        "rating_max": int(df["rating"].max()),
        "rating_median": float(df["rating"].median()),
        "rating_std": float(df["rating"].std()),
    }
    return result
