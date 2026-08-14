"""Data loading, time splitting, risk sets, and the once-per-pair constraint.

All time handling is in raw Unix seconds unless a module-level ``t0`` offset
is applied.  The audit guarantees of this SNAP release:

- every directed user pair appears at most once (no within-dyad revisions),
- no self-loops,
- timestamps from 2010-11-08 to 2016-01-25 UTC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_PATH = "data/bitcoin_otc/soc-sign-bitcoinotc.csv.gz"
COLUMNS = ["source", "target", "rating", "timestamp"]


def load_events(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw event table (already audited elsewhere, we keep it light)."""
    df = pd.read_csv(
        path,
        names=COLUMNS,
        dtype={"source": np.int64, "target": np.int64, "rating": np.int64},
        compression="gzip",
    )
    return df


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Stable sort by timestamp; ties broken by insertion order (row order)."""
    return df.sort_values("timestamp", kind="stable", ignore_index=True)


def node_entry_times(df: pd.DataFrame) -> pd.DataFrame:
    """First appearance (as source or target) of each user, in original time."""
    t0 = df["timestamp"].min()
    first_as_source = df.groupby("source")["timestamp"].min().rename("first")
    first_as_target = df.groupby("target")["timestamp"].min().rename("first")
    first = pd.concat([first_as_source, first_as_target], axis=1)
    entry = first.min(axis=1).rename("entry").rename_axis("node")
    return entry.to_frame().sort_values("entry")


@dataclass
class TimeSplit:
    """A chronological split of the event stream."""

    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: float  # seconds (raw); train = events with timestamp <= cutoff

    @classmethod
    def by_quantile(cls, df: pd.DataFrame, q: float = 0.7) -> "TimeSplit":
        df = sort_by_time(df)
        cutoff = df["timestamp"].quantile(q)
        return cls(
            train=df[df["timestamp"] <= cutoff],
            test=df[df["timestamp"] > cutoff],
            cutoff=float(cutoff),
        )

    @classmethod
    def by_fraction_of_span(cls, df: pd.DataFrame, f: float = 0.7) -> "TimeSplit":
        """Cut at fraction ``f`` of the observed time span, not of events."""
        df = sort_by_time(df)
        t0, t1 = df["timestamp"].min(), df["timestamp"].max()
        cutoff = t0 + f * (t1 - t0)
        return cls(
            train=df[df["timestamp"] <= cutoff],
            test=df[df["timestamp"] > cutoff],
            cutoff=float(cutoff),
        )


def expanding_windows(
    df: pd.DataFrame,
    n_windows: int = 5,
    train_fraction: float = 0.4,
    test_fraction: float = 0.12,
) -> list[TimeSplit]:
    """Rolling validation: expanding training, non-overlapping test blocks.

    Windows are defined on the sorted timestamp grid.  Each window uses all
    events up to its train cutoff, and evaluates on the following block.
    """
    df = sort_by_time(df)
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    span = t1 - t0
    starts = t0 + train_fraction * span
    test_len = test_fraction * span
    out = []
    for k in range(n_windows):
        train_cut = starts + k * test_len
        test_cut = train_cut + test_len
        if test_cut >= t1 - 1.0:
            break
        out.append(
            TimeSplit(
                train=df[df["timestamp"] <= train_cut],
                test=df[(df["timestamp"] > train_cut) & (df["timestamp"] <= test_cut)],
                cutoff=float(train_cut),
            )
        )
    return out


def active_nodes(df: pd.DataFrame, t: float) -> np.ndarray:
    """Nodes that have appeared (as source or target) at or before ``t``.

    These are the members of the risk set at time ``t``.
    """
    nodes = np.unique(np.concatenate([df[df["timestamp"] <= t]["source"].values,
                                      df[df["timestamp"] <= t]["target"].values]))
    return nodes


def at_risk_pairs_at(active: np.ndarray, used: set | None = None) -> np.ndarray:
    """All ordered pairs (i, j), i != j, among ``active``, minus ``used`` pairs.

    Returns (N_active*(N_active-1), 2) int array of candidate directed pairs.
    For N_active > ~5000 this is memory-heavy; callers should subsample or
    aggregate rather than materialise the full risk set.
    """
    n = len(active)
    if n * (n - 1) > 2**25:  # > 33.5M pairs: refuse to materialise silently
        raise MemoryError(f"risk set too large: {n * (n - 1)} ordered pairs")
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pairs = np.stack([active[i.ravel()], active[j.ravel()]], axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # no self-loops
    if used:
        used_arr = np.array(sorted(used), dtype=np.int64)
        # keep pairs not in used (symmetric membership check is overkill;
        # the data audit guarantees used pairs are directed and unique)
        key = pairs[:, 0] * 10_000_000 + pairs[:, 1]
        used_key = used_arr[:, 0] * 10_000_000 + used_arr[:, 1]
        pairs = pairs[~np.isin(key, used_key)]
    return pairs
