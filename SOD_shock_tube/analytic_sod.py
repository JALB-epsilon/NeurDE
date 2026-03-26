import math
import numpy as np


def _sound_speed(state, gamma):
    return math.sqrt(gamma * state["p"] / state["rho"])


def _pressure_function(p_star, state, gamma):
    rho_k = state["rho"]
    p_k = state["p"]
    a_k = _sound_speed(state, gamma)

    if p_star > p_k:
        a_coeff = 2.0 / ((gamma + 1.0) * rho_k)
        b_coeff = (gamma - 1.0) / (gamma + 1.0) * p_k
        root = math.sqrt(a_coeff / (p_star + b_coeff))
        f_val = (p_star - p_k) * root
        f_der = root * (1.0 - 0.5 * (p_star - p_k) / (p_star + b_coeff))
        return f_val, f_der

    exponent = (gamma - 1.0) / (2.0 * gamma)
    pressure_ratio = p_star / p_k
    f_val = 2.0 * a_k / (gamma - 1.0) * (pressure_ratio ** exponent - 1.0)
    f_der = (pressure_ratio ** (-(gamma + 1.0) / (2.0 * gamma))) / (rho_k * a_k)
    return f_val, f_der


def _solve_star_region(left_state, right_state, gamma, max_iter=50, tol=1e-10):
    p_old = max(1e-8, 0.5 * (left_state["p"] + right_state["p"]))

    for _ in range(max_iter):
        f_left, fd_left = _pressure_function(p_old, left_state, gamma)
        f_right, fd_right = _pressure_function(p_old, right_state, gamma)
        residual = f_left + f_right + right_state["u"] - left_state["u"]
        derivative = fd_left + fd_right
        p_new = max(1e-8, p_old - residual / derivative)
        if abs(p_new - p_old) < tol:
            p_old = p_new
            break
        p_old = p_new

    f_left, _ = _pressure_function(p_old, left_state, gamma)
    f_right, _ = _pressure_function(p_old, right_state, gamma)
    u_star = 0.5 * (left_state["u"] + right_state["u"] + f_right - f_left)
    return p_old, u_star


def _sample_left_state(xi, left_state, p_star, u_star, gamma):
    rho_l = left_state["rho"]
    u_l = left_state["u"]
    p_l = left_state["p"]
    a_l = _sound_speed(left_state, gamma)

    if p_star > p_l:
        pressure_ratio = p_star / p_l
        shock_speed = u_l - a_l * math.sqrt(
            1.0 + (gamma + 1.0) / (2.0 * gamma) * (pressure_ratio - 1.0)
        )
        if xi <= shock_speed:
            return rho_l, u_l, p_l
        rho_star = rho_l * (
            (pressure_ratio + (gamma - 1.0) / (gamma + 1.0))
            / (((gamma - 1.0) / (gamma + 1.0)) * pressure_ratio + 1.0)
        )
        return rho_star, u_star, p_star

    a_star = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
    head_speed = u_l - a_l
    tail_speed = u_star - a_star
    if xi <= head_speed:
        return rho_l, u_l, p_l
    if xi >= tail_speed:
        rho_star = rho_l * (p_star / p_l) ** (1.0 / gamma)
        return rho_star, u_star, p_star

    u = 2.0 / (gamma + 1.0) * (a_l + 0.5 * (gamma - 1.0) * u_l + xi)
    a = 2.0 / (gamma + 1.0) * (a_l + 0.5 * (gamma - 1.0) * (u_l - xi))
    rho = rho_l * (a / a_l) ** (2.0 / (gamma - 1.0))
    p = p_l * (a / a_l) ** (2.0 * gamma / (gamma - 1.0))
    return rho, u, p


def _sample_right_state(xi, right_state, p_star, u_star, gamma):
    rho_r = right_state["rho"]
    u_r = right_state["u"]
    p_r = right_state["p"]
    a_r = _sound_speed(right_state, gamma)

    if p_star > p_r:
        pressure_ratio = p_star / p_r
        shock_speed = u_r + a_r * math.sqrt(
            1.0 + (gamma + 1.0) / (2.0 * gamma) * (pressure_ratio - 1.0)
        )
        if xi >= shock_speed:
            return rho_r, u_r, p_r
        rho_star = rho_r * (
            (pressure_ratio + (gamma - 1.0) / (gamma + 1.0))
            / (((gamma - 1.0) / (gamma + 1.0)) * pressure_ratio + 1.0)
        )
        return rho_star, u_star, p_star

    a_star = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
    head_speed = u_r + a_r
    tail_speed = u_star + a_star
    if xi >= head_speed:
        return rho_r, u_r, p_r
    if xi <= tail_speed:
        rho_star = rho_r * (p_star / p_r) ** (1.0 / gamma)
        return rho_star, u_star, p_star

    u = 2.0 / (gamma + 1.0) * (-a_r + 0.5 * (gamma - 1.0) * u_r + xi)
    a = 2.0 / (gamma + 1.0) * (a_r - 0.5 * (gamma - 1.0) * (u_r - xi))
    rho = rho_r * (a / a_r) ** (2.0 / (gamma - 1.0))
    p = p_r * (a / a_r) ** (2.0 * gamma / (gamma - 1.0))
    return rho, u, p


def exact_riemann_solution(x, t, left_state, right_state, gamma, x0):
    rho = np.empty_like(x, dtype=np.float64)
    velocity = np.empty_like(x, dtype=np.float64)
    pressure = np.empty_like(x, dtype=np.float64)

    if t <= 0:
        left_mask = x <= x0
        right_mask = ~left_mask
        rho[left_mask] = left_state["rho"]
        velocity[left_mask] = left_state["u"]
        pressure[left_mask] = left_state["p"]
        rho[right_mask] = right_state["rho"]
        velocity[right_mask] = right_state["u"]
        pressure[right_mask] = right_state["p"]
        return rho, velocity, pressure

    p_star, u_star = _solve_star_region(left_state, right_state, gamma)
    xi = (x - x0) / t

    left_mask = xi <= u_star
    for idx in np.where(left_mask)[0]:
        rho[idx], velocity[idx], pressure[idx] = _sample_left_state(
            float(xi[idx]),
            left_state,
            p_star,
            u_star,
            gamma,
        )

    for idx in np.where(~left_mask)[0]:
        rho[idx], velocity[idx], pressure[idx] = _sample_right_state(
            float(xi[idx]),
            right_state,
            p_star,
            u_star,
            gamma,
        )

    return rho, velocity, pressure


def analytic_reference_from_case(case_number, case_params, steps):
    riemann = case_params.get("riemann")
    if riemann is None:
        raise ValueError(f"Case {case_number} does not define Riemann states in Sod_cases_param.yml.")

    x = np.arange(case_params["X"], dtype=np.float64)
    x0 = float(riemann.get("x0", 0.5 * (case_params["X"] - 1)))
    gamma = float(case_params["vuy"])
    left_state = riemann["left"]
    right_state = riemann["right"]
    y_dim = int(case_params["Y"])

    all_rho = []
    all_ux = []
    all_temp = []
    all_pressure = []

    for step in range(1, steps + 1):
        rho_1d, ux_1d, pressure_1d = exact_riemann_solution(
            x=x,
            t=float(step),
            left_state=left_state,
            right_state=right_state,
            gamma=gamma,
            x0=x0,
        )
        temp_1d = pressure_1d / rho_1d
        all_rho.append(np.tile(rho_1d[None, :], (y_dim, 1)))
        all_ux.append(np.tile(ux_1d[None, :], (y_dim, 1)))
        all_temp.append(np.tile(temp_1d[None, :], (y_dim, 1)))
        all_pressure.append(np.tile(pressure_1d[None, :], (y_dim, 1)))

    analytic_rho = np.stack(all_rho)
    analytic_ux = np.stack(all_ux)
    analytic_temp = np.stack(all_temp)
    analytic_pressure = np.stack(all_pressure)
    return analytic_rho, analytic_ux, analytic_temp, analytic_pressure
