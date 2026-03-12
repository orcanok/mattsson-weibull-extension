import numpy as np

_S_ERROR = "Bisection failed to bracket root for s(x)."


def u_crra(w: np.ndarray, rho: float, theta: float, eps: float) -> np.ndarray:
    """Computes CRRA utility from wage income.

    Utility matches the specification used in Mattsson & Weibull:
      - For rho != 1:
          u(w) = theta * (((w + eps)^(1 - rho) - 1) / (1 - rho))
      - For rho == 1:
          u(w) = theta * log(w + eps)

    Args:
        w: Wage (scalar or array). Evaluated elementwise. Typically w >= 0.
        rho: Relative risk aversion parameter (rho >= 0). rho = 0 corresponds to
            risk neutrality (linear utility up to an additive constant).
        theta: Positive scale parameter (theta > 0).
        eps: Positive shift/external income parameter (eps > 0) ensuring w + eps
            is positive and log is well-defined at w = 0.

    Returns:
        np.ndarray of utility values with the same shape as `w` (float dtype).
    """
    w = np.asarray(w, dtype=float)
    if rho == 1.0:
        return theta * np.log(w + eps)
    return theta * (((w + eps) ** (1.0 - rho) - 1.0) / (1.0 - rho))


def softmax_from_util(p0: np.ndarray, util: np.ndarray) -> tuple[np.ndarray, float]:
    """Computes the exponential-tilt (softmax) distribution and normalizer.

    Given baseline probabilities `p0` and utility/index values `util`, this
    function returns
        p_k = p0_k * exp(util_k) / Z,
    where
        Z = sum_k p0_k * exp(util_k).

    In the KL-divergence model, `log(Z)` corresponds to the agent's indirect
    utility.

    Args:
        p0: Array-like baseline probability vector. Must be nonnegative and the
            same length as `util`.
        util: Array-like utility/index vector. Must be the same length as `p0`.

    Returns:
        A tuple (p, Z) where:
          - p: np.ndarray of softmax probabilities summing to 1 (up to floating
            point error).
          - Z: float normalizing constant `sum(p0 * exp(util))`.

    """
    weights = p0 * np.exp(util)
    Z = float(weights.sum())
    return weights / Z, Z


def s_share_scalar(
    x: float,
    rho: float,
    theta: float,
    eps: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Solves for the sharing rule s(x) under CRRA utility.

    Computes s >= 0 that satisfies the implicit equation:
        x = s + (1/theta) * ((s + eps)**rho - eps**rho).

    Special cases:
      - rho = 0: s(x) = x
      - rho = 1: s(x) = (theta / (1 + theta)) * x

    Args:
        x: Nonnegative surplus level.
        rho: Relative risk aversion parameter (rho >= 0).
        theta: Positive scale parameter (theta > 0).
        eps: Positive shift/external income parameter (eps > 0).
        tol: Absolute tolerance for the root-finding stopping rule.
        max_iter: Maximum number of bisection iterations.

    Returns:
        The solution s(x) as a float.
    """
    if x <= 0.0:
        return 0.0
    if rho == 0.0:
        return x
    if rho == 1.0:
        return (theta / (1.0 + theta)) * x

    # monotone equation in s: f(s)= s + (1/theta)((s+eps)^rho - eps^rho) - x = 0
    def f(s: float) -> float:
        return s + (1.0 / theta) * ((s + eps) ** rho - (eps**rho)) - x

    lo, hi = 0.0, x  # s(x) <= x always
    flo = f(lo)  # = -x < 0
    fhi = f(hi)  # should be >= 0
    if fhi < 0:
        # extremely rare numerically; expand hi until feasible
        hi = max(1.0, x)
        for _ in range(60):
            fhi = f(hi)
            if fhi >= 0:
                break
            hi *= 2.0
        if fhi < 0:
            raise RuntimeError(_S_ERROR)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) <= tol:
            return mid
        if fmid > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def s_share(x: np.ndarray, rho: float, theta: float, eps: float) -> np.ndarray:
    """Applies the sharing rule s(x) elementwise to an array.

    Args:
        x: Array of nonnegative surplus values.
        rho: Relative risk aversion parameter (rho >= 0).
        theta: Positive scale parameter (theta > 0).
        eps: Positive shift/external income parameter (eps > 0).

    Returns:
        Array of the same shape as `x`, where each entry is s_share_scalar(x_ij, ...).
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        out[it.multi_index] = s_share_scalar(float(it[0]), rho, theta, eps)
        it.iternext()
    return out


def wage_from_threshold(
    pi: np.ndarray, t: float, rho: float, theta: float, eps: float
) -> np.ndarray:
    """Computes canonical threshold--sharing wages w(pi; t) = s((pi - t)_+).

    Args:
        pi: Array of gross profit outcomes (support points).
        t: Threshold (face value) parameter. Outcomes below t yield zero surplus.
        rho: Relative risk aversion parameter (rho >= 0) used in the sharing rule.
        theta: Positive scale parameter (theta > 0) used in the sharing rule.
        eps: External income parameter (eps > 0) used in the sharing rule.

    Returns:
        Array of wages with the same shape as `pi`, computed as s(max(pi - t, 0)).
    """
    x = np.maximum(0.0, pi - t)
    return s_share(x, rho, theta, eps)


def ubar_two_period_from_index(u_index: float, delta: float) -> float:
    """Computes the two-period outside option from a utility-level index.

    Uses the convention that the one-period reservation utility is:
        ubar^(1) = ln(u_index),
    and the two-period reservation utility (evaluated at period 1) equals:
        ubar^(2) = (1 + delta) * ubar^(1).

    Args:
        u_index: Positive index value used to define the one-period outside option.
        delta: Agent's discount factor for period 2 utility (0 < delta <= 1).

    Returns:
        The two-period reservation utility ubar^(2) as a float.
    """
    ubar_1 = float(np.log(u_index))
    return (1.0 + float(delta)) * ubar_1
