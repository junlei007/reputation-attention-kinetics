"""Numerical solver for the kinetic (macroscopic) capital equation.

The mean-field limit of the multiplicative microscopic process
(notes/WP3_kinetic_derivation.md) is, for the type coordinate u on a finite
block structure (K groups) and capital c in R:

  d/dt f_t(u, dc) = -decay(delta) - loss(kappa_t(u,c)) + jump gain

with per-capita target rate

  kappa_t(u,c) = A_t(u) * psi2(c),
  A_t(u)       = sum_v a(v) W(v,u) S_t(v),   S_t(v) = int psi1(c') f_t(v, dc').

The solver is a deterministic structure-preserving finite-volume scheme on a
u x c grid:

  1. decay step: exact multiplicative remap (backward semi-Lagrangian with
     linear interpolation of the cell-averaged density);
  2. jump step: for each discrete mark r in the support of the mark kernel,
     mass flows from cell m' to the fractional index of c_{m'} + eta*r,
     with rate kappa(c_{m'}) and mark probability q(r | c_{m'});
  3. per-u-cell renormalisation to conserve the cell population.

For marks that depend on the target capital only, the per-cell mark
distribution is static and precomputed.  Sender-dependent marks (the general
case) would require averaging against the sender field; the API accepts a
mark kernel callable ``q(r, c)`` that the caller may update per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KineticSpec:
    """Block-structured macroscopic specification."""

    K: int                       # number of latent groups
    pi: np.ndarray               # (K,) mass proportion per group (sums to 1)
    activity: np.ndarray         # (K,) group activity a_k
    W: np.ndarray                # (K,K) block kernel
    delta: float                 # capital decay rate
    eta: float = 0.1             # jump scale
    psi1: callable = lambda c: 1.0          # bounded, nonnegative, Lipschitz
    psi2: callable = lambda c: 1.0          # bounded, nonnegative, Lipschitz
    # mark kernel: callable r_support, q_probs per capital cell.
    # NOTE: q_cells columns must be ordered consistently with r_values.
    r_values: np.ndarray | None = None      # (R,) discrete mark support
    q_cells: np.ndarray | None = None       # (M, R) q(r | c_m), rows sum 1
    c_min: float = -2.0
    c_max: float = 2.0
    M: int = 2000               # capital cells
    dt_max: float = 0.05
    T: float = 4.0
    initial: str = "delta0"     # "delta0": all mass at c=0 (MC convention)
    source_rate: np.ndarray | None = None  # (K,) exogenous entry rate (mass at c=0)


class KineticSolver:
    """Deterministic finite-volume solver for the block kinetic equation."""

    def __init__(self, spec: KineticSpec):
        self.s = spec
        self.c_min = spec.c_min
        self.c_max = spec.c_max
        self.c_grid = np.linspace(spec.c_min, spec.c_max, spec.M + 1)
        self.c_mid = 0.5 * (self.c_grid[:-1] + self.c_grid[1:])
        self.h = self.c_grid[1] - self.c_grid[0]
        # (K, M) mass density f[k, m]; mass = f * h (proportion of population)
        self.f = np.zeros((spec.K, spec.M))
        if spec.initial == "delta0":
            m0 = int(np.argmin(np.abs(self.c_mid)))
            self.f[:, m0] = spec.pi / self.h
        else:
            for k in range(spec.K):
                self.f[k, :] = spec.pi[k] / (spec.c_max - spec.c_min)
        self.t = 0.0
        self._marks = self._precompute_marks()

    def _precompute_marks(self):
        """Destination fractional indices and probabilities per (m, r)."""
        s = self.s
        if s.r_values is None:
            # default: uniform magnitude 1..10, sign from a flat logistic
            r = np.array([sgn * mag / 10.0
                          for sgn in (-1, 1) for mag in range(1, 11)])
            q = np.full((s.M, len(r)), 0.05)   # sign/magnitude independent
            return r, q, self._destinations(r)
        return s.r_values, s.q_cells, self._destinations(s.r_values)

    def _destinations(self, r_values):
        """For each (m, r): fractional cell index of c_m + eta*r."""
        idx = (self.c_mid[None, :] + self.s.eta * r_values[:, None]
               - self.c_min) / self.h
        lo = np.floor(idx).astype(int)
        w = idx - lo
        lo = np.clip(lo, 0, self.s.M - 2)
        return lo, w   # shapes (R, M)

    # ------------------------------------------------------------------
    @staticmethod
    def _eval_psi(fn, c_mid: np.ndarray) -> np.ndarray:
        v = np.asarray(fn(c_mid), dtype=float)
        if v.size == 1:
            return np.full_like(c_mid, float(v))
        return v

    def _sender_field(self) -> np.ndarray:
        """S_t(k) = int psi1(c) f_t(k, dc)  (proportions, per group)."""
        v = self._eval_psi(self.s.psi1, self.c_mid)
        return (self.f @ v) * self.h

    def _rate(self) -> np.ndarray:
        """kappa_t(k, m) = A_t(k) * psi2(c_m), with

        A_t(k) = sum_l W[l, k] * a_l * S_t(l)   (senders l -> targets k)."""
        S = self._sender_field()
        A = self.s.W.T @ (self.s.activity * S)   # (K,)
        v2 = self._eval_psi(self.s.psi2, self.c_mid)
        return A[:, None] * v2[None, :]

    def _decay_step(self, dt: float) -> None:
        """Conservative decay remap (exact for piecewise-linear f).

        Mass in cell m at t+dt is the integral of f over the stretched
        preimage interval [g_{m-1}*e^{delta dt}, g_m*e^{delta dt}]; the
        antiderivative F of the piecewise-linear density is interpolated
        exactly at those points.  The remap conserves mass to machine
        precision (the antiderivative differences ARE the preimage masses;
        no stretch-factor division is applied, which would lose mass at
        rate 1 - e^{-(delta+rho) dt} per step and only appear conserved
        under renormalisation).  For delta = 0 the map is the identity
        (no numerical diffusion, no mean drift)."""
        s = self.s
        scale = np.exp(s.delta * dt)
        # antiderivative of the piecewise-linear density through midpoints,
        # sampled at grid edges (exact trapezoid sums)
        F = np.concatenate(
            [np.zeros((s.K, 1)), np.cumsum(self.f, axis=1) * self.h], axis=1)
        pts = self.c_grid * scale                   # (M+1,) stretched edges
        idx = (pts - s.c_min) / self.h
        lo = np.clip(np.floor(idx).astype(int), 0, s.M - 1)
        w = np.clip(idx - lo, 0.0, 1.0)
        F_at = F[:, lo] * (1 - w) + F[:, lo + 1] * w
        mass_new = np.diff(F_at, axis=1)           # mass per cell at t+dt
        self.f = mass_new / self.h
        # mass that would leave the grid is folded into the boundary cell;
        # the per-group renormalisation restores exact conservation

    def _jump_step(self, dt: float) -> None:
        """Marked jumps: survival exp(-kappa dt), gain from all (m', r)
        destinations; the two combine to conserve mass to O((kappa dt)^2)."""
        s = self.s
        kap = self._rate()
        self.f *= np.exp(-kap * dt)       # exact survival for the frozen rate
        lo, w = self._marks_dest
        q = self._marks_q                  # (M, R)
        src = self.f * kap * dt            # (K, M) expected jumps per cell
        for r_i in range(len(self._marks_r)):
            dest_lo = lo[r_i]              # (M,) fractional dest of each cell
            wgt = w[r_i]
            contrib = src * q[:, r_i][None, :]     # (K, M)
            self._add_scatter(contrib * (1 - wgt), dest_lo)
            self._add_scatter(contrib * wgt, dest_lo + 1)

    def _add_scatter(self, contrib: np.ndarray, dest: np.ndarray) -> None:
        """contrib (K, M) scattered into self.f at column indices dest (M,)."""
        np.add.at(self.f, (np.arange(self.s.K)[:, None], dest[None, :]), contrib)

    def _renormalise(self) -> None:
        """Restore per-group total mass (pi_k) lost to grid edges / numerics."""
        tot = self.f.sum(axis=1) * self.h
        self.f *= self.s.pi[:, None] / np.maximum(tot, 1e-300)[:, None]

    # ------------------------------------------------------------------
    def step(self) -> float:
        s = self.s
        kap = self._rate()
        max_k = float(kap.max()) if kap.size else 0.0
        dt = min(s.dt_max, 0.1 / max(max_k, 1e-12))
        self._decay_step(dt)
        self._jump_step(dt)
        if s.source_rate is not None:
            # exogenous entries: mass enters at the c=0 cell; the per-group
            # mass then grows beyond pi, so renormalisation is skipped
            m0 = int(np.argmin(np.abs(self.c_mid)))
            self.f[:, m0] += s.source_rate * dt / self.h
        else:
            self._renormalise()
        self.t += dt
        return dt

    def run(self, T: float | None = None) -> np.ndarray:
        """Evolve to T; returns the final density (K, M)."""
        T = self.s.T if T is None else T
        while self.t < T - 1e-12:
            self.step()
        return self.f

    @property
    def _marks_r(self):
        return self._marks[0]

    @property
    def _marks_q(self):
        return self._marks[1]

    @property
    def _marks_dest(self):
        return self._marks[2]

    # ------------------------------------------------------------------
    def capital_marginal(self) -> np.ndarray:
        """Aggregate capital distribution (mass per cell, sums to 1)."""
        return self.f.sum(axis=0) * self.h

    def group_marginals(self) -> np.ndarray:
        return self.f * self.h   # (K, M) probabilities

    def aggregate_rate(self) -> float:
        return float(((self.f * self._rate()) * self.h).sum())

    def event_rate_trajectory(self, n_bins: int = 30) -> np.ndarray:
        """Mean per-capita event rate in each of ``n_bins`` time bins.
        Call on a FRESH solver instance (evolves its state to T)."""
        s = self.s
        T = s.T
        rates = np.zeros(n_bins)
        idx = 0
        boundary = T / n_bins
        while self.t < T - 1e-12:
            dt = self.step()
            while self.t >= (idx + 1) * boundary and idx < n_bins - 1:
                idx += 1
            rates[idx] += self.aggregate_rate() * dt
        return rates / (T / n_bins)


# --------------------------------------------------------------------------
# two-dimensional (capital, attention) solver: joint law on R x R_+
# --------------------------------------------------------------------------

@dataclass
class KineticSpec2D:
    """Two-dimensional block kinetic specification (v0.11 model).

    The limiting law is f_t(k, c, h) on K types x capital cells x attention
    cells (h >= 0, mass proportion per type pi_k).  The per-capita target
    rate factorises as kappa_t(k, c, h) = A_t(k) psi_C(c) psi_H(h) with
    A_t(k) = sum_l pi_l a_l W[l, k] S_t(l),  S_t(l) = int psi1(x) f_t(l, dx).
    """

    K: int
    pi: np.ndarray               # (K,) mass proportion per group
    activity: np.ndarray         # (K,) group activity a_k
    W: np.ndarray                # (K,K) block kernel
    delta: float                 # capital decay rate
    rho: float                   # attention decay rate (0 = static H)
    eta: float = 0.1             # jump scale
    psi1: callable = lambda c: 1.0      # sender response (C-only instance)
    psiC: callable = lambda c: 1.0      # target capital response
    psiH: callable = lambda h: 1.0      # target attention response
    r_values: np.ndarray | None = None  # (R,) discrete mark support
    q_cells: np.ndarray | None = None   # (M_c, R) q(r | c_m), rows sum 1
    c_min: float = -2.0
    c_max: float = 2.0
    M_c: int = 400
    H_max: float = 16.0
    M_h: int = 160
    dt_max: float = 0.05
    T: float = 4.0
    initial: str = "delta0"     # delta0: all mass at (c=0, h=0)


class KineticSolver2D:
    """Deterministic finite-volume solver for the joint (C, H) kinetic law.

    Grid: c cells (uniform on [c_min, c_max]) x h cells (uniform on
    [0, H_max] with dh such that the +1 attention jump is an exact integer
    cell offset).  The decay step is a conservative remap applied separably
    along c (scale e^{delta dt}) and h (scale e^{rho dt}); the jump step is
    an explicit marked transport with survival exp(-kappa dt).  Per-group
    mass is restored at every step by renormalisation; the mass drift
    before renormalisation is tracked in ``self.raw_drift``.
    """

    def __init__(self, spec: KineticSpec2D):
        self.s = spec
        self.c_min = spec.c_min
        self.c_max = spec.c_max
        self.c_grid = np.linspace(spec.c_min, spec.c_max, spec.M_c + 1)
        self.c_mid = 0.5 * (self.c_grid[:-1] + self.c_grid[1:])
        self.dc = self.c_grid[1] - self.c_grid[0]
        self.h_grid = np.linspace(0.0, spec.H_max, spec.M_h + 1)
        self.h_mid = 0.5 * (self.h_grid[:-1] + self.h_grid[1:])
        self.dh = self.h_grid[1] - self.h_grid[0]
        # the +1 attention jump must be an exact integer cell offset
        self.h_units = int(round(1.0 / self.dh))
        assert abs(self.h_units * self.dh - 1.0) < 1e-9, \
            "dh must divide 1.0 exactly (attention jump = +1)"
        # (K, M_c, M_h) mass density per cell
        self.f = np.zeros((spec.K, spec.M_c, spec.M_h))
        if spec.initial == "delta0":
            m0 = int(np.argmin(np.abs(self.c_mid)))
            self.f[:, m0, 0] = spec.pi / (self.dc * self.dh)
        else:
            for k in range(spec.K):
                self.f[k, :, :] = spec.pi[k] / (
                    (spec.c_max - spec.c_min) * spec.H_max)
        self.t = 0.0
        self.raw_drift = 0.0          # max rel. per-group drift before renorm
        self._marks_r, self._marks_q = self._precompute_marks()

    def _precompute_marks(self):
        s = self.s
        if s.r_values is None:
            r = np.array([sgn * mag / 10.0
                          for sgn in (-1, 1) for mag in range(1, 11)])
            q = np.full((s.M_c, len(r)), 0.05)
            return r, q
        return s.r_values, np.asarray(s.q_cells)

    # ------------------------------------------------------------------
    @staticmethod
    def _eval_psi(fn, x_mid: np.ndarray) -> np.ndarray:
        v = np.asarray(fn(x_mid), dtype=float)
        if v.size == 1:
            return np.full_like(x_mid, float(v))
        return v

    def _sender_field(self) -> np.ndarray:
        """S_t(k) = sum_{c,h} psi1(c) f(k, c, h) dc dh."""
        v = self._eval_psi(self.s.psi1, self.c_mid)
        return ((self.f * v[None, :, None]).sum(axis=(1, 2))) * self.dc * self.dh

    def _rate(self) -> np.ndarray:
        """kappa_t(k, c, h) = A_t(k) psi_C(c) psi_H(h)."""
        s = self.s
        S = self._sender_field()
        A = s.W.T @ (s.activity * S)          # (K,)
        vC = self._eval_psi(s.psiC, self.c_mid)
        vH = self._eval_psi(s.psiH, self.h_mid)
        return A[:, None, None] * vC[None, :, None] * vH[None, None, :]

    # -- decay: separable conservative remap ----------------------------
    @staticmethod
    def _remap_1d(f: np.ndarray, axis: int, scale: float,
                  offset: float = 0.0) -> np.ndarray:
        """Mass-conserving stretch remap along one axis by factor scale
        (piecewise-linear antiderivative interpolation).

        The mass in cell j at t+dt is the integral of f over the stretched
        preimage interval, whose unit-edge indices are
        j*scale + offset.  For the capital axis (c_min != 0) the offset is
        c_min*(scale - 1)/dc; for the attention axis (h_min = 0) it is 0.
        """
        M = f.shape[axis]
        F = np.cumsum(f, axis=axis)
        # antiderivative F on the edges (mass so far, with an extra edge
        # index along ``axis``)
        F = np.concatenate([np.zeros(f.shape[:axis]
                                     + (1,)
                                     + f.shape[axis + 1:]), F], axis=axis)
        # stretched edge positions in unit-edge index
        j_idx = np.arange(M + 1, dtype=float) * scale + offset
        lo = np.floor(j_idx).astype(int)
        w = j_idx - lo
        lo = np.clip(lo, 0, M - 1)
        hi = np.clip(lo + 1, 0, M)
        if axis == 1:
            F_lo = F[:, lo, :]
            F_hi = F[:, hi, :]
            F_at = F_lo * (1 - w)[None, :, None] + F_hi * w[None, :, None]
        else:
            F_lo = F[:, :, lo]
            F_hi = F[:, :, hi]
            F_at = F_lo * (1 - w)[None, None, :] + F_hi * w[None, None, :]
        # the antiderivative differences ARE the preimage masses: no
        # division by scale (which would lose mass at rate 1 - 1/scale per
        # step, only masked by renormalisation)
        mass_new = np.diff(F_at, axis=axis)
        return mass_new

    def _decay_step(self, dt: float) -> None:
        s = self.s
        if s.delta > 0:
            off = s.c_min * (np.exp(s.delta * dt) - 1.0) / self.dc
            self.f = self._remap_1d(self.f, axis=1, scale=np.exp(s.delta * dt),
                                    offset=off)
        if s.rho > 0:
            self.f = self._remap_1d(self.f, axis=2, scale=np.exp(s.rho * dt))

    # -- jump: explicit marked transport ---------------------------------
    def _jump_step(self, dt: float) -> None:
        s = self.s
        kap = self._rate()
        self.f *= np.exp(-kap * dt)              # exact survival (frozen rate)
        src = self.f * kap * dt                  # (K, Mc, Mh)
        r_vals, q = self._marks_r, self._marks_q  # q: (Mc, R)
        M_c, M_h = s.M_c, s.M_h
        hu = self.h_units
        for r_i, rv in enumerate(r_vals):
            qv = q[:, r_i][None, :, None]        # (1, Mc, 1)
            contrib = src * qv
            # c-direction: fractional shift by eta*rv/dc cells
            shift = s.eta * rv / self.dc
            i0 = int(np.floor(shift))
            wfrac = shift - i0
            if wfrac > 0:
                target = (1 - wfrac) * self._shift_c(contrib, i0, M_c)
                target += wfrac * self._shift_c(contrib, i0 + 1, M_c)
            else:
                target = self._shift_c(contrib, i0, M_c)
            # h-direction: exact integer shift of +1 (clip at H_max)
            if hu > 0:
                out = np.zeros_like(target)
                out[:, :, hu:] = target[:, :, :-hu]
                out[:, :, -1] += target[:, :, -hu:].sum(axis=2)
            else:
                out = target
            self.f += out

    @staticmethod
    def _shift_c(x: np.ndarray, i: int, M_c: int) -> np.ndarray:
        """Fractional cell shift along the c axis, clipping at the edges."""
        if i == 0:
            return x
        out = np.zeros_like(x)
        if i > 0:
            out[:, i:, :] = x[:, :-i, :]
            out[:, -1, :] += x[:, -i:, :].sum(axis=1)   # clip at c_max
        else:
            out[:, :i, :] = x[:, -i:, :]
            out[:, 0, :] += x[:, :-i, :].sum(axis=1)    # clip at c_min
        return out

    # ------------------------------------------------------------------
    def _renormalise(self) -> None:
        tot = self.f.sum(axis=(1, 2)) * self.dc * self.dh
        rel = np.max(np.abs(tot - self.s.pi) / np.maximum(self.s.pi, 1e-300))
        self.raw_drift = max(self.raw_drift, float(rel))
        self.f *= self.s.pi[:, None, None] / \
            np.maximum(tot, 1e-300)[:, None, None]

    def step(self) -> float:
        kap = self._rate()
        max_k = float(kap.max()) if kap.size else 0.0
        dt = min(self.s.dt_max, 0.1 / max(max_k, 1e-12))
        self._decay_step(dt)
        self._jump_step(dt)
        self._renormalise()
        self.t += dt
        return dt

    def run(self, T: float | None = None) -> np.ndarray:
        T = self.s.T if T is None else T
        while self.t < T - 1e-12:
            self.step()
        return self.f

    # ------------------------------------------------------------------
    def marginal_c(self) -> np.ndarray:
        """Capital marginal (mass per c-cell, sums to 1)."""
        return self.f.sum(axis=(0, 2)) * self.dc * self.dh

    def marginal_h(self) -> np.ndarray:
        """Attention marginal (mass per h-cell, sums to 1)."""
        return self.f.sum(axis=(0, 1)) * self.dc * self.dh

    def group_marginals(self) -> np.ndarray:
        return self.f * self.dc * self.dh

    def aggregate_rate(self) -> float:
        return float(((self.f * self._rate()) * self.dc * self.dh).sum())

    def event_rate_trajectory(self, n_bins: int = 30) -> np.ndarray:
        """Mean per-capita event rate in each of ``n_bins`` time bins.
        Call on a FRESH solver instance (evolves its state to T)."""
        s = self.s
        T = s.T
        rates = np.zeros(n_bins)
        idx = 0
        boundary = T / n_bins
        while self.t < T - 1e-12:
            dt = self.step()
            while self.t >= (idx + 1) * boundary and idx < n_bins - 1:
                idx += 1
            rates[idx] += self.aggregate_rate() * dt
        return rates / (T / n_bins)
