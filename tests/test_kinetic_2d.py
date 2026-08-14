"""Unit tests for the two-dimensional kinetic solver.

Focus: mass conservation of the decay remap (regression test for the
stretch-factor division bug: dividing the antiderivative differences by
scale lost mass at rate 1 - e^{-(delta+rho) dt} per step, which was only
masked by renormalisation), and the analytic shrink of a point mass under
pure decay.
"""

import numpy as np
import pytest

from dengyunetwork.kinetic import KineticSpec2D, KineticSolver2D


def _pure_decay_solver(delta, rho, M_c=400, M_h=160, H_max=16.0, dt=0.005):
    """Solver with W = 0 (no jumps): pure two-axis decay."""
    spec = KineticSpec2D(
        K=1, pi=np.array([1.0]), activity=np.zeros(1), W=np.zeros((1, 1)),
        delta=delta, rho=rho, eta=0.1,
        M_c=M_c, H_max=H_max, M_h=M_h, dt_max=dt, T=2.0)
    return KineticSolver2D(spec)


def test_decay_only_mass_conservation_to_machine_precision():
    s = _pure_decay_solver(delta=np.log(2) / 30.0, rho=0.5)
    s.f[:] = 0.0
    mc = int(np.argmin(np.abs(s.c_mid - 0.5)))
    mh = int(np.argmin(np.abs(s.h_mid - 3.0)))
    s.f[0, mc, mh] = 1.0 / (s.dc * s.dh)
    s.run()
    tot = (s.f * s.dc * s.dh).sum()
    # the remap itself conserves mass; the residual is interpolation/boundary
    # error only, not the O((delta+rho) dt) stretch-factor loss
    assert abs(tot - 1.0) < 1e-9, f"decay-only mass drift {abs(tot - 1.0):.2e}"


@pytest.mark.slow
def test_decay_only_point_mass_shrinks_analytically():
    delta, rho, T = np.log(2.0) / 5.0, 0.4, 2.0
    s = _pure_decay_solver(delta=delta, rho=rho, M_c=1000, M_h=400,
                           H_max=20.0, dt=0.01)  # dh = 0.05: divides 1.0
    c0, h0 = 0.5, 3.0
    mc = int(np.argmin(np.abs(s.c_mid - c0)))
    mh = int(np.argmin(np.abs(s.h_mid - h0)))
    s.f[:] = 0.0
    s.f[0, mc, mh] = 1.0 / (s.dc * s.dh)
    s.run(T)
    cm = s.marginal_c()          # mass per cell (sums to 1)
    hm = s.marginal_h()
    c_mean = float((cm * s.c_mid).sum())
    h_mean = float((hm * s.h_mid).sum())
    assert abs(c_mean - c0 * np.exp(-delta * T)) < 0.02
    assert abs(h_mean - h0 * np.exp(-rho * T)) < 0.1


def test_decay_only_raw_drift_is_not_the_stretch_factor_loss():
    s = _pure_decay_solver(delta=np.log(2) / 30.0, rho=0.5, dt=0.005)
    s.run()
    # per-step stretch-factor loss would accumulate to ~1 - e^{-(delta+rho)T}
    # = ~0.66; the true drift is at interpolation/boundary level
    assert s.raw_drift < 1e-4, f"raw drift {s.raw_drift:.2e}"


def test_full_run_mass_conservation():
    """With jumps as well, per-group mass is conserved to the renorm level;
    the raw drift before renormalisation is the jump-step time-discretisation
    error O((kappa dt)^2) (survival exp(-kappa dt) plus the jump gain do not
    cancel exactly per step), not the decay-remap error of the regression."""
    spec = KineticSpec2D(
        K=2, pi=np.array([0.6, 0.4]), activity=np.array([1.0, 1.5]),
        W=np.ones((2, 2)), delta=0.05, rho=0.3, eta=0.1, T=2.0)
    s = KineticSolver2D(spec)
    s.run()
    tot = (s.f * s.dc * s.dh).sum(axis=(1, 2))
    assert np.allclose(tot, spec.pi, atol=1e-9)
    assert s.raw_drift < 5e-3   # jump-step discretisation error, not remap loss
