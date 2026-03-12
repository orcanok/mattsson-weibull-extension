from typing import Any

import numpy as np
import optimagic as om

from .model import s_share, softmax_from_util, u_crra, wage_from_threshold


def two_period_given_thresholds(
    pi: np.ndarray,
    p0: np.ndarray,
    beta: float,
    delta: float,
    rho: float,
    theta: float,
    eps: float,
    t1: float,
    t2_vec: np.ndarray,
) -> dict[str, Any]:
    """Evaluates the two-period model under canonical threshold parameters.

    This function takes a 3-outcome calibration and threshold parameters
    (t1, t2(1), t2(2), t2(3)) and computes:
      - the induced period-2 wages and softmax distributions for each history,
      - the induced period-1 wages and softmax distribution,
      - the agent's ex-ante utility U = ln(D),
      - the principal's expected discounted profit Pi.

    The canonical wage schedules are:
      - Period 2: w2(i, j) = s(max(pi_j - t2(i), 0))
      - Period 1: w1(i)    = s(max(pi_i + C_i - t1, 0))
    where C_i = beta * E[pi_2 - w2 | history i] is the discounted continuation
    profit and s(·) is the sharing rule implied by CRRA/KL.

    Args:
        pi: Array of gross profit outcomes, shape (3,).
        p0: Baseline/default probabilities over outcomes, shape (3,). Typically
            sums to 1.
        beta: Principal's discount factor in (0, 1].
        delta: Agent's discount factor in (0, 1].
        rho: Relative risk aversion parameter (rho >= 0).
        theta: Positive scale parameter (theta > 0) in CRRA utility.
        eps: Positive shift/external income parameter (eps > 0).
        t1: Period-1 threshold (face value) parameter.
        t2_vec: Array of period-2 thresholds by history, shape (3,). Entry i is
            the threshold t2(i) used after observing outcome i in period 1.

    Returns:
        Dictionary with the following keys:
          - "t1": float, the input t1.
          - "t2": np.ndarray, shape (3,), the input t2_vec.
          - "w1": np.ndarray, shape (3,), period-1 wages.
          - "w2": np.ndarray, shape (3, 3), period-2 wages by history and outcome.
          - "p1": np.ndarray, shape (3,), induced period-1 distribution.
          - "p2": np.ndarray, shape (3, 3), induced period-2 distributions by history.
          - "Z2": np.ndarray, shape (3,), period-2 normalizers Z_i.
          - "U2": np.ndarray, shape (3,), continuation utilities ln(Z_i).
          - "U": float, ex-ante utility ln(D).
          - "Pi_i": np.ndarray, shape (3,), principal profit conditional on history.
          - "Pi": float, principal expected discounted profit.
          - "tildePi": np.ndarray, shape (3,), undiscounted continuation profits.
          - "C": np.ndarray, shape (3,), discounted continuation profits beta*tildePi.
          - "D": float, period-1 normalizer D.
    """
    t2_vec = np.asarray(t2_vec, dtype=float).reshape(3)

    # Period 2 per history i
    w2 = np.zeros((3, 3), dtype=float)
    p2 = np.zeros((3, 3), dtype=float)
    Z2 = np.zeros(3, dtype=float)
    U2 = np.zeros(3, dtype=float)
    tildePi = np.zeros(3, dtype=float)  # principal continuation profit (undiscounted)
    C = np.zeros(3, dtype=float)  # discounted continuation profit

    for i in range(3):
        w2[i, :] = wage_from_threshold(pi, float(t2_vec[i]), rho, theta, eps)
        util2 = u_crra(w2[i, :], rho, theta, eps)
        p2[i, :], Z2[i] = softmax_from_util(p0, util2)
        U2[i] = float(np.log(Z2[i]))
        tildePi[i] = float(np.sum(p2[i, :] * (pi - w2[i, :])))
        C[i] = beta * tildePi[i]

    # Period 1 wages: w1_i = s((pi_i + C_i - t1)_+)
    x1 = np.maximum(0.0, (pi + C) - t1)
    w1 = s_share(x1, rho, theta, eps)
    util1 = u_crra(w1, rho, theta, eps)

    # p1 weights: p0_i exp(u(w1_i)) Z2_i^delta
    weights1 = p0 * np.exp(util1) * (Z2**delta)
    D = float(weights1.sum())
    p1 = weights1 / D
    U = float(np.log(D))

    # Principal profit conditional on history i: (pi_i - w1_i) + beta*tildePi_i
    Pi_i = (pi - w1) + beta * tildePi
    Pi = float(np.sum(p1 * Pi_i))

    return {
        "t1": float(t1),
        "t2": t2_vec.copy(),
        "w1": w1,
        "w2": w2,
        "p1": p1,
        "p2": p2,
        "Z2": Z2,
        "U2": U2,
        "U": U,
        "Pi_i": Pi_i,
        "Pi": Pi,
        "tildePi": tildePi,
        "C": C,
        "D": D,
    }


def make_objective(pi, p0, beta, delta, rho, theta, eps):
    """Creates an optimagic objective for the two-period threshold problem.

    The returned function maps a dict of named parameters to a scalar criterion.
    It is designed to be passed to `optimagic.minimize`. The criterion equals
    minus the principal's expected discounted profit, so minimizing it is
    equivalent to maximizing profit.

    Expected parameter names in `params` are:
      - "t1": period-1 threshold
      - "t21", "t22", "t23": period-2 thresholds for histories i=1,2,3

    Args:
        pi: Array of gross profit outcomes, shape (3,).
        p0: Baseline/default probabilities, shape (3,).
        beta: Principal's discount factor in (0, 1].
        delta: Agent's discount factor in (0, 1].
        rho: Relative risk aversion parameter (rho >= 0).
        theta: Positive scale parameter (theta > 0).
        eps: Positive shift/external income parameter (eps > 0).

    Returns:
        A callable `obj(params)` that returns a float criterion to minimize.
    """

    def obj(params):
        # params is a dict if we use optimagic's named parameters
        t1 = params["t1"]
        t2 = np.array([params["t21"], params["t22"], params["t23"]], dtype=float)

        out = two_period_given_thresholds(pi, p0, beta, delta, rho, theta, eps, t1, t2)
        return -out["Pi"]

    return obj


def make_pc_constraint(pi, p0, beta, delta, rho, theta, eps, ubar):
    """Creates a participation constraint U(params) >= ubar for optimagic.

    The constraint is evaluated on the agent's ex-ante utility returned by
    `two_period_given_thresholds`. It can be passed to `optimagic.minimize`
    via the `constraints=[...]` argument.

    Expected parameter names in `params` are:
      - "t1": period-1 threshold
      - "t21", "t22", "t23": period-2 thresholds for histories i=1,2,3

    Args:
        pi: Array of gross profit outcomes, shape (3,).
        p0: Baseline/default probabilities, shape (3,).
        beta: Principal's discount factor in (0, 1].
        delta: Agent's discount factor in (0, 1].
        rho: Relative risk aversion parameter (rho >= 0).
        theta: Positive scale parameter (theta > 0).
        eps: Positive shift/external income parameter (eps > 0).
        ubar: Reservation utility level to be satisfied by the contract.

    Returns:
        An `optimagic.NonlinearConstraint` enforcing `U(params) >= ubar`.
    """

    def U_of_params(params):
        t1 = params["t1"]
        t2 = np.array([params["t21"], params["t22"], params["t23"]], dtype=float)
        out = two_period_given_thresholds(pi, p0, beta, delta, rho, theta, eps, t1, t2)
        return out["U"]  # scalar

    return om.NonlinearConstraint(
        func=U_of_params,
        lower_bound=ubar,
        upper_bound=None,
    )


def find_feasible_start(
    pi, p0, beta, delta, rho, theta, eps, ubar, bounds, seed=0, max_tries=20000
):
    """Finds a feasible initial guess for constrained optimization.

    Randomly samples threshold parameters within the provided bounds until it
    finds a point satisfying the participation constraint U >= ubar, where U is
    the agent's ex-ante utility computed by `two_period_given_thresholds`.

    The returned dict is compatible with optimagic's named-parameter interface.

    Args:
        pi: Array of gross profit outcomes, shape (3,).
        p0: Baseline/default probabilities, shape (3,).
        beta: Principal's discount factor in (0, 1].
        delta: Agent's discount factor in (0, 1].
        rho: Relative risk aversion parameter (rho >= 0).
        theta: Positive scale parameter (theta > 0).
        eps: Positive shift/external income parameter (eps > 0).
        ubar: Reservation utility level; feasibility requires U >= ubar.
        bounds: optimagic.Bounds object with keys "t1", "t21", "t22", "t23".
        seed: Seed for the random number generator used for sampling.
        max_tries: Maximum number of random samples before giving up.

    Returns:
        A dict with keys "t1", "t21", "t22", "t23" representing a feasible set of
        thresholds within `bounds`.

    Raises:
        RuntimeError: If no feasible point is found after `max_tries` samples.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds.lower, bounds.upper
    for _ in range(max_tries):
        params = {
            "t1": rng.uniform(lo["t1"], hi["t1"]),
            "t21": rng.uniform(lo["t21"], hi["t21"]),
            "t22": rng.uniform(lo["t22"], hi["t22"]),
            "t23": rng.uniform(lo["t23"], hi["t23"]),
        }
        t2 = np.array([params["t21"], params["t22"], params["t23"]], float)
        out = two_period_given_thresholds(
            pi, p0, beta, delta, rho, theta, eps, params["t1"], t2
        )
        if out["U"] >= ubar:
            return params
    raise RuntimeError("Could not find feasible start; expand bounds or lower ubar.")


def make_threshold_bounds(
    pi,
    margin_mult_t2: float = 2.0,
    margin_mult_t1: float = 3.0,
    extra_shift_t1: float = 0.0,
    extra_shift_t2: float = 0.0,
):
    """Builds data-driven bounds for the threshold parameters (t1, t2(1:3)).

    The bounds are constructed from the scale of the outcome support `pi`. Let
    R = max(pi) - min(pi). Then:
      - t2(i) bounds extend the outcome range by `margin_mult_t2 * R`,
      - t1 bounds extend the outcome range by `margin_mult_t1 * R` (typically
        wider because t1 applies to pi + continuation terms),
    and both can be additionally widened by the optional shifts.

    Args:
        pi: Array-like outcome/payoff support (gross profits), shape (3,).
        margin_mult_t2: Multiple of the outcome range used to extend the bounds
            for each t2(i) beyond [min(pi), max(pi)].
        margin_mult_t1: Multiple of the outcome range used to extend the bounds
            for t1 beyond [min(pi), max(pi)].
        extra_shift_t1: Optional additive widening (absolute units) for t1 bounds.
        extra_shift_t2: Optional additive widening (absolute units) for t2 bounds.

    Returns:
        optimagic.Bounds object with keys "t1", "t21", "t22", "t23".
    """
    pi = np.asarray(pi, dtype=float).reshape(3)
    pi_min, pi_max = float(pi.min()), float(pi.max())
    R = pi_max - pi_min
    if R <= 0:
        # Degenerate support; still return something sensible.
        R = max(1.0, abs(pi_max))

    # t2 bounds: thresholds relative to period-2 output scale
    t2_low = pi_min - margin_mult_t2 * R - extra_shift_t2
    t2_high = pi_max + margin_mult_t2 * R + extra_shift_t2

    # t1 bounds: compares to pi + continuation profit, so slightly wider
    t1_low = pi_min - margin_mult_t1 * R - extra_shift_t1
    t1_high = pi_max + margin_mult_t1 * R + extra_shift_t1

    lower = {"t1": t1_low, "t21": t2_low, "t22": t2_low, "t23": t2_low}
    upper = {"t1": t1_high, "t21": t2_high, "t22": t2_high, "t23": t2_high}

    return om.Bounds(lower=lower, upper=upper)


def solve_with_optimagic(pi, p0, beta, delta, rho, theta, eps, ubar):
    """Solves the two-period threshold problem using optimagic.

    Sets up and solves the constrained optimization problem over threshold
    parameters (t1, t2(1), t2(2), t2(3)):

      maximize   Pi(t1, t2)
      subject to U(t1, t2) >= ubar
                 (t1, t2) within bounds

    The objective Pi and utility U are computed by `two_period_given_thresholds`
    under the canonical threshold--sharing wage structure. Optimization is
    performed with `optimagic.minimize` using SciPy's SLSQP algorithm and a
    multistart strategy.

    Args:
        pi: Array of gross profit outcomes, shape (3,).
        p0: Baseline/default probabilities, shape (3,).
        beta: Principal's discount factor in (0, 1].
        delta: Agent's discount factor in (0, 1].
        rho: Relative risk aversion parameter (rho >= 0).
        theta: Positive scale parameter (theta > 0).
        eps: Positive shift/external income parameter (eps > 0).
        ubar: Reservation utility level for the participation constraint.

    Returns:
        A tuple (res, out_star) where:
          - res: optimagic optimization result object.
          - out_star: dict produced by `two_period_given_thresholds` evaluated at
            the optimizer's parameter values, containing thresholds, wages,
            induced distributions, utility, and profit.
    """
    obj = make_objective(pi, p0, beta, delta, rho, theta, eps)
    pc = make_pc_constraint(pi, p0, beta, delta, rho, theta, eps, ubar)

    bounds = make_threshold_bounds(pi, margin_mult_t2=2.0, margin_mult_t1=3.0)

    # Initial guess (can be anything reasonable)
    x0 = {"t1": 0.0, "t21": 0.3, "t22": 0.3, "t23": 0.3}
    # x0 = find_feasible_start(pi, p0, beta, delta, rho, theta, eps, ubar, bounds, seed=1)
    # Local method: derivative-free is safer with kinks
    # (Powell / Nelder-Mead are common choices)
    res = om.minimize(
        fun=obj,
        params=x0,
        algorithm="scipy_slsqp",
        bounds=bounds,
        constraints=[pc],
        multistart=om.MultistartOptions(n_samples=30, seed=1),
    )

    # Extract best thresholds and compute full objects
    x_star = res.params
    t1_star = x_star["t1"]
    t2_star = np.array([x_star["t21"], x_star["t22"], x_star["t23"]], dtype=float)
    out_star = two_period_given_thresholds(
        pi, p0, beta, delta, rho, theta, eps, t1_star, t2_star
    )
    return res, out_star
