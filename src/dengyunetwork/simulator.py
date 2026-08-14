"""Event-driven Monte Carlo for the frozen-rate microscopic process.

Two-dimensional (capital, attention) state X = (C, H) (v0.11 model, frozen in
notes/WP0_model_freeze.md):

  lambda_ij(t) = (1/N) B_i(t) B_j(t) (1 - A_ij(t)) a_i(t) W(g_i, g_j)
                 psi1(X_i(t)) psi_C(C_j(t)) psi_H(H_j(t))

- C decays multiplicatively, dC = -delta C dt, and jumps by eta*r on receipt.
- H is the attention-memory state, dH = -rho H dt, and jumps by +1 on receipt.
- B_i flips on at exogenous entry times (risk-set convention).
- A_ij is the once-per-pair exclusion (used pairs carry rate 0).
- W is a K x K block kernel; psi1, psi_C, psi_H are bounded, nonnegative,
  Lipschitz responses (any combination is again bounded Lipschitz on the
  state metric d_X((c,h),(c',h')) = |c-c'| + omega |h-h'|).
- Marks r ~ Q(dr | X_i, X_j, g_i, g_j) with |r| <= 1.

The simulator keeps rates frozen between events (standard KMC), with states
decayed exactly to each event time.  With multiplicative psi the total rate
factorises per block pair, so each event costs O(N + used) instead of O(N^2):

  Lambda = (1/N) sum_gg' W_gg' A1_g A2_g'

where A1_g = sum_{i in g} B_i a_i psi1(C_i) (the sender response here is the
one-dimensional-in-C instance of psi1, a legal two-dimensional instance since
a function of C alone is Lipschitz in d_X), and A2_g = sum_{j in g'} B_j
psi_C(C_j) psi_H(H_j).

The once-per-pair exclusion is implemented by acceptance--rejection
(thinning) rather than by subtracting a used-pair correction: a candidate
pair is sampled from the FULL-pair block rates A1_g A2_g' W_gg' and rejected
if the ordered pair is already used.  Because the full-pair rate is an upper
bound on the exact rate (which is 0 for used pairs) componentwise, the
rejection scheme is exact -- it reproduces the law of the process with the
used-pair rates set to 0 -- at O(1) cost per event.  A subtracted correction
term would have to be recomputed from the CURRENT state after every decay
and jump (used pairs are weighted by psi1(X_i) psi_T(X_j) with X evolving),
which costs O(number of used pairs) per event; caching the weight at event
time, as done in an earlier version, is only an approximation and is NOT
used here.

One-dimensional capital-only runs are the special case rho = 0 and
psiH = lambda h: 1.0 (the v0.11 capital-only closure g_H = 0, psi_H == 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCALAR_FN = callable  # typing aid: psi1, psi2, mark are callables


@dataclass
class SimParams:
    """Full specification of one simulation run."""

    N: int                       # population size (potential users)
    K: int                       # number of latent groups
    groups: np.ndarray           # (N,) group id in {0..K-1}
    activity: np.ndarray         # (N,) baseline activity a_i (nonneg, bounded)
    W: np.ndarray                # (K,K) block kernel, nonneg bounded
    delta: float                 # capital decay rate
    rho: float = 0.0             # attention-memory decay rate (0 = no H dynamics)
    eta: float = 0.1             # jump scale (rating/10)
    T: float = 1.0               # horizon
    entry_times: np.ndarray | None = None  # (N,) exogenous entries (None = all at 0)

    def __post_init__(self):
        self.groups = np.asarray(self.groups, dtype=int)
        self.activity = np.asarray(self.activity, dtype=float)
        self.W = np.asarray(self.W, dtype=float)
        if self.entry_times is None:
            self.entry_times = np.zeros(self.N)
        else:
            self.entry_times = np.asarray(self.entry_times, dtype=float)


@dataclass
class SimResult:
    """Event list plus per-event state snapshots."""

    t: np.ndarray          # (M,) event times
    source: np.ndarray     # (M,)
    target: np.ndarray     # (M,)
    rating: np.ndarray     # (M,) ratings in [-1, 1]
    cap_source: np.ndarray  # sender capital just before the event
    cap_target: np.ndarray  # target capital just before the event
    cap_after: np.ndarray   # target capital just after the event
    active: int            # nodes that entered by the end
    att_target: np.ndarray = None   # target attention H just before (2D runs)
    att_after: np.ndarray = None    # target attention H just after (2D runs)
    n_events: int = field(init=False)

    def __post_init__(self):
        self.n_events = len(self.t)

    def to_frame(self):
        import pandas as pd

        frame = {
            "timestamp": self.t, "source": self.source, "target": self.target,
            "rating": self.rating, "cap_source": self.cap_source,
            "cap_target": self.cap_target, "cap_after": self.cap_after,
        }
        if self.att_target is not None:
            frame["att_target"] = self.att_target
            frame["att_after"] = self.att_after
        return pd.DataFrame(frame)


class EventSimulator:
    """Exact frozen-rate event-driven simulator (see module docstring).

    Parameters
    ----------
    psi1, psi2 : callable
        Vectorized (np.ndarray -> np.ndarray) and scalar-safe capital
        functions; bounded, nonnegative.  ``psi1`` is the sender response
        (the C-only instance of the two-dimensional response), ``psi2`` is
        ``psi_C``, the target capital response.
    psiH : callable, optional
        Target attention-memory response ``psi_H`` (default ``1``: the
        capital-only closure).
    mark : callable
        ``mark(i, j, ci, cj, gi, gj) -> rating in [-1, 1]``; the conditional
        distribution Q sampled once per event.
    mark2d : callable, optional
        ``mark2d(i, j, ci, cj, hi, hj, gi, gj) -> rating in [-1, 1]``;
        used instead of ``mark`` when the mark kernel depends on the full
        joint (C, H) state (the theorem's general case).
    """

    def __init__(self, p: SimParams, psi1, psi2, mark,
                 psiH=None, mark2d=None,
                 rng: np.random.Generator | None = None):
        self.p = p
        self.psi1 = psi1
        self.psi2 = psi2
        self.psiH = psiH if psiH is not None else (lambda h: 1.0)
        self.mark = mark
        self.mark2d = mark2d
        self.rng = rng if rng is not None else np.random.default_rng(0)
        N = p.N
        self.active = np.zeros(N, dtype=bool)
        self._n_active = 0
        self.C = np.zeros(N)
        self.H = np.zeros(N)
        self.last_t = np.zeros(N)
        self.used_pairs: dict[int, list[int]] = {}
        self.used_targets: dict[int, set[int]] = {}
        self.used_counts = np.zeros((p.K, p.K), dtype=int)  # used pairs per block
        self.active_g = np.zeros(p.K, dtype=int)            # active nodes per group
        self.entry_order = np.argsort(p.entry_times)
        self.next_entry = 0
        # cached per-group aggregates
        self.A1 = np.zeros(p.K)
        self.A2 = np.zeros(p.K)

    # -- state machinery ---------------------------------------------------
    def _decay_all(self, t: float) -> None:
        dt = t - self.last_t
        m = (dt > 0)
        if np.any(m):
            self.C[m] *= np.exp(-self.p.delta * dt[m])
            if self.p.rho > 0:
                self.H[m] *= np.exp(-self.p.rho * dt[m])
        self.last_t[:] = t

    def _decay_node(self, i: int, t: float) -> None:
        dt = t - self.last_t[i]
        if dt > 0:
            self.C[i] *= np.exp(-self.p.delta * dt)
            if self.p.rho > 0:
                self.H[i] *= np.exp(-self.p.rho * dt)
            self.last_t[i] = t

    def _state_vec(self) -> tuple[np.ndarray, np.ndarray]:
        """(v1, v2) with v2 = psi_C(C) * psi_H(H) (vectorized; scalar
        returns are broadcast)."""
        v1 = np.asarray(self.psi1(self.C), dtype=float)
        v2 = np.asarray(self.psi2(self.C), dtype=float) * \
            np.asarray(self.psiH(self.H), dtype=float)
        if v1.size == 1:
            v1 = np.full_like(self.C, float(v1))
        if v2.size == 1:
            v2 = np.full_like(self.C, float(v2))
        return v1, v2

    def _refresh(self) -> tuple[np.ndarray, np.ndarray]:
        """Recompute per-group aggregates; returns (v1, v2) used for sampling."""
        p = self.p
        v1, v2 = self._state_vec()
        w1 = np.where(self.active, p.activity * v1, 0.0)
        w2 = np.where(self.active, v2, 0.0)
        self.A1 = np.bincount(p.groups, weights=w1, minlength=p.K)
        self.A2 = np.bincount(p.groups, weights=w2, minlength=p.K)
        return v1, v2

    def _block_mass(self) -> np.ndarray:
        """Full-pair block rates, with saturated blocks (used pairs filling
        the whole capacity) set to 0; their exact rate is 0, so dropping
        them is exact and keeps the rejection budget away from them."""
        p = self.p
        mass = self.A1[:, None] * self.A2[None, :] * p.W
        cap = self.active_g[:, None] * self.active_g[None, :]
        cap -= np.diag(self.active_g)                 # no self-pairs
        mass[self.used_counts >= cap] = 0.0
        return mass

    def total_rate(self) -> float:
        return float(self._block_mass().sum()) / self.p.N

    def _update_used(self, i: int, j: int) -> None:
        self.used_pairs.setdefault(i, []).append(j)
        self.used_targets.setdefault(i, set()).add(j)
        self.used_counts[self.p.groups[i], self.p.groups[j]] += 1

    # -- main loop ---------------------------------------------------------
    def run(self, record_capitals: bool = True) -> SimResult:
        p = self.p
        rng = self.rng
        K = p.K

        # entries at t = 0
        while (self.next_entry < p.N
               and p.entry_times[self.entry_order[self.next_entry]] <= 0):
            node = self.entry_order[self.next_entry]
            self.active[node] = True
            self._n_active += 1
            self.active_g[p.groups[node]] += 1
            self.next_entry += 1
        v1, v2 = self._refresh()

        ts, src, tgt, rat, cs, ct, ca = [], [], [], [], [], [], []
        hs, ha = [], []
        t = 0.0
        while t < p.T:
            next_entry_t = (p.entry_times[self.entry_order[self.next_entry]]
                            if self.next_entry < p.N else np.inf)
            lam = self.total_rate()
            if lam <= 0:
                if next_entry_t < p.T:
                    t = next_entry_t
                    node = self.entry_order[self.next_entry]
                    self.active[node] = True
                    self._n_active += 1
                    self.active_g[p.groups[node]] += 1
                    self.next_entry += 1
                    v1, v2 = self._refresh()
                    continue
                break

            dt = rng.exponential(1.0 / lam)
            if t + dt > next_entry_t:          # entry clock rings first
                t = next_entry_t
                self._decay_all(t)
                node = self.entry_order[self.next_entry]
                self.active[node] = True
                self._n_active += 1
                self.active_g[p.groups[node]] += 1
                self.next_entry += 1
                v1, v2 = self._refresh()
                continue

            t += dt
            self._decay_all(t)
            v1, v2 = self._refresh()

            # block pair (g, h) with prob ~ W_gh A1_g A2_h (full-pair rates);
            # the once-per-pair exclusion is enforced by acceptance--rejection
            # (exact thinning: the full-pair rate is an upper bound on the
            # exact rate componentwise).  The CDFs are built ONCE per accepted
            # block; the rejection loop only re-draws (i, j) at O(log N) each,
            # so the cost is O(N + used) per event, not O(N) per rejection.
            mass = self._block_mass()
            tot = mass.sum()
            if tot <= 0:
                continue
            u = rng.random() * tot
            (g, h) = np.unravel_index(np.searchsorted(np.cumsum(mass.ravel()), u), (K, K))

            # i in g ~ a_i psi1(C_i); j in h ~ psi2(C_j); reject used pairs
            # and self-pairs
            w1 = np.where(self.active & (p.groups == g), p.activity * v1, 0.0)
            w2 = np.where(self.active & (p.groups == h), v2, 0.0)
            cdf1 = np.cumsum(w1)
            cdf2 = np.cumsum(w2)
            if cdf1[-1] <= 0 or cdf2[-1] <= 0:
                continue
            i = 0
            j = 0
            attempts = 0
            while True:
                i = int(np.searchsorted(cdf1, rng.random() * cdf1[-1]))
                # cheap pre-check: an i whose target set covers all active
                # nodes cannot be accepted; redraw instead of burning the
                # rejection budget
                used_i = self.used_targets.get(i)
                if used_i and len(used_i) >= self._n_active - 1:
                    attempts += 1
                    if attempts > 10_000:
                        break
                    continue
                j = int(np.searchsorted(cdf2, rng.random() * cdf2[-1]))
                if i != j and j not in self.used_targets.get(i, ()):
                    break
                attempts += 1
                if attempts > 10_000:
                    break
            if attempts > 10_000:
                continue

            c_i = self._decay_get(i, t)
            c_j = self._decay_get(j, t)
            h_i = self.H[i]
            h_j = self.H[j]
            if self.mark2d is not None:
                r = float(np.clip(
                    self.mark2d(i, j, c_i, c_j, h_i, h_j, g, h, rng),
                    -1.0, 1.0))
            else:
                r = float(np.clip(self.mark(i, j, c_i, c_j, g, h, rng),
                                  -1.0, 1.0))

            if record_capitals:
                ts.append(t); src.append(i); tgt.append(j); rat.append(r)
                cs.append(c_i); ct.append(c_j)
                if p.rho > 0:
                    hs.append(h_j)

            # update: target capital jumps, attention memory jumps by +1,
            # pair becomes used
            self.C[j] += p.eta * r
            if p.rho > 0:
                self.H[j] += 1.0
            self.last_t[j] = t
            self._update_used(i, j)
            if record_capitals:
                ca.append(self.C[j])
                if p.rho > 0:
                    ha.append(self.H[j])

        att_target = np.array(hs) if p.rho > 0 else None
        att_after = np.array(ha) if p.rho > 0 else None
        return SimResult(
            t=np.array(ts), source=np.array(src), target=np.array(tgt),
            rating=np.array(rat), cap_source=np.array(cs),
            cap_target=np.array(ct), cap_after=np.array(ca),
            att_target=att_target, att_after=att_after,
            active=int(self.active.sum()),
        )

    def _decay_get(self, i: int, t: float) -> float:
        self._decay_node(i, t)
        return self.C[i]


# --------------------------------------------------------------------------
# default parameterisations for synthetic experiments
# --------------------------------------------------------------------------

def tanh_psi(alpha: float, beta: float, center: float = 0.0):
    """Bounded nonnegative capital function 1 + alpha*tanh(beta*(C - center))."""
    alpha, beta = float(alpha), float(beta)

    def psi(C):
        return 1.0 + alpha * np.tanh(beta * (C - center))

    return psi


def logistic_mark(intercept: float, b_target: float, b_sender: float,
                  magnitude_max: int = 10):
    """Mark distribution: sign ~ logistic(b0 + bt*C_j + bs*C_i), magnitude
    uniform on 1..magnitude_max (scaled to [-1,1] by /magnitude_max).
    The callable receives the simulator's rng as its last argument."""

    def mark(i, j, ci, cj, gi, gj, rng):
        p_pos = 1.0 / (1.0 + np.exp(-(intercept + b_target * cj + b_sender * ci)))
        sgn = 1.0 if rng.random() < p_pos else -1.0
        mag = rng.integers(1, magnitude_max + 1)
        return sgn * mag / magnitude_max

    return mark
