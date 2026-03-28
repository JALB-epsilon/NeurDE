import math
from typing import Dict, Tuple

import numpy as np
import torch

_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}
_NUMPY_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
}


def resolve_numpy_dtype(dtype):
    if isinstance(dtype, np.dtype):
        dtype_name = dtype.name
    elif isinstance(dtype, type) and issubclass(dtype, np.generic):
        dtype_name = np.dtype(dtype).name
    else:
        dtype_name = str(dtype).lower()
    if dtype_name not in _NUMPY_DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {sorted(_NUMPY_DTYPE_MAP)}")
    return _NUMPY_DTYPE_MAP[dtype_name]


def resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        if dtype not in _TORCH_DTYPE_MAP.values():
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        return dtype
    dtype_name = str(dtype).lower()
    if dtype_name not in _TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {sorted(_TORCH_DTYPE_MAP)}")
    return _TORCH_DTYPE_MAP[dtype_name]


def _sound_speed(rho: float, p: float, gamma: float) -> float:
    return math.sqrt(gamma * p / rho)


def _pressure_function(p: float, rho_k: float, p_k: float, a_k: float, gamma: float) -> Tuple[float, float]:
    if p <= p_k:
        pressure_ratio = p / p_k
        expo = (gamma - 1.0) / (2.0 * gamma)
        f = (2.0 * a_k / (gamma - 1.0)) * (pressure_ratio**expo - 1.0)
        fd = (1.0 / (rho_k * a_k)) * pressure_ratio ** (-(gamma + 1.0) / (2.0 * gamma))
        return f, fd

    a_coeff = 2.0 / ((gamma + 1.0) * rho_k)
    b_coeff = ((gamma - 1.0) / (gamma + 1.0)) * p_k
    root = math.sqrt(a_coeff / (p + b_coeff))
    f = (p - p_k) * root
    fd = root * (1.0 - 0.5 * (p - p_k) / (p + b_coeff))
    return f, fd


def _solve_star_pressure(left: Dict[str, float], right: Dict[str, float], gamma: float, tol: float = 1.0e-8) -> float:
    rho_l, u_l, p_l = left["rho"], left["u"], left["p"]
    rho_r, u_r, p_r = right["rho"], right["u"], right["p"]
    a_l = _sound_speed(rho_l, p_l, gamma)
    a_r = _sound_speed(rho_r, p_r, gamma)

    p_pv = 0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (a_l + a_r)
    p_old = max(tol, p_pv)

    for _ in range(50):
        f_l, fd_l = _pressure_function(p_old, rho_l, p_l, a_l, gamma)
        f_r, fd_r = _pressure_function(p_old, rho_r, p_r, a_r, gamma)
        residual = f_l + f_r + u_r - u_l
        p_new = p_old - residual / (fd_l + fd_r)
        if p_new < tol:
            p_new = tol
        if abs(p_new - p_old) / (0.5 * (p_new + p_old)) < tol:
            return p_new
        p_old = p_new

    return p_old


def _star_velocity(p_star: float, left: Dict[str, float], right: Dict[str, float], gamma: float) -> float:
    a_l = _sound_speed(left["rho"], left["p"], gamma)
    a_r = _sound_speed(right["rho"], right["p"], gamma)
    f_l, _ = _pressure_function(p_star, left["rho"], left["p"], a_l, gamma)
    f_r, _ = _pressure_function(p_star, right["rho"], right["p"], a_r, gamma)
    return 0.5 * (left["u"] + right["u"] + f_r - f_l)


def solve_exact_riemann(x: np.ndarray, t: float, left: Dict[str, float], right: Dict[str, float], gamma: float, x0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    rho_l, u_l, p_l = left["rho"], left["u"], left["p"]
    rho_r, u_r, p_r = right["rho"], right["u"], right["p"]

    if t <= 0.0:
        mask = x <= x0
        rho = np.where(mask, rho_l, rho_r)
        u = np.where(mask, u_l, u_r)
        p = np.where(mask, p_l, p_r)
        return rho, u, p

    a_l = _sound_speed(rho_l, p_l, gamma)
    a_r = _sound_speed(rho_r, p_r, gamma)
    p_star = _solve_star_pressure(left, right, gamma)
    u_star = _star_velocity(p_star, left, right, gamma)

    rho_star_l = rho_l * (
        (p_star / p_l + (gamma - 1.0) / (gamma + 1.0))
        / (((gamma - 1.0) / (gamma + 1.0)) * (p_star / p_l) + 1.0)
    ) if p_star > p_l else rho_l * (p_star / p_l) ** (1.0 / gamma)
    rho_star_r = rho_r * (
        (p_star / p_r + (gamma - 1.0) / (gamma + 1.0))
        / (((gamma - 1.0) / (gamma + 1.0)) * (p_star / p_r) + 1.0)
    ) if p_star > p_r else rho_r * (p_star / p_r) ** (1.0 / gamma)

    xi = (x - x0) / t
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)

    left_mask = xi <= u_star
    xi_left = xi[left_mask]

    if p_star > p_l:
        s_l = u_l - a_l * math.sqrt(((gamma + 1.0) / (2.0 * gamma)) * (p_star / p_l) + (gamma - 1.0) / (2.0 * gamma))
        left_state_mask = xi_left <= s_l
        star_mask = ~left_state_mask
        rho[left_mask] = np.where(left_state_mask, rho_l, rho_star_l)
        u[left_mask] = np.where(left_state_mask, u_l, u_star)
        p[left_mask] = np.where(left_state_mask, p_l, p_star)
    else:
        s_head = u_l - a_l
        a_star_l = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
        s_tail = u_star - a_star_l
        left_state_mask = xi_left <= s_head
        star_mask = xi_left >= s_tail
        fan_mask = (~left_state_mask) & (~star_mask)

        rho_values = np.empty_like(xi_left)
        u_values = np.empty_like(xi_left)
        p_values = np.empty_like(xi_left)

        rho_values[left_state_mask] = rho_l
        u_values[left_state_mask] = u_l
        p_values[left_state_mask] = p_l

        rho_values[star_mask] = rho_star_l
        u_values[star_mask] = u_star
        p_values[star_mask] = p_star

        if np.any(fan_mask):
            xi_fan = xi_left[fan_mask]
            u_fan = (2.0 / (gamma + 1.0)) * (a_l + 0.5 * (gamma - 1.0) * u_l + xi_fan)
            a_fan = (2.0 / (gamma + 1.0)) * (a_l + 0.5 * (gamma - 1.0) * (u_l - xi_fan))
            rho_values[fan_mask] = rho_l * (a_fan / a_l) ** (2.0 / (gamma - 1.0))
            u_values[fan_mask] = u_fan
            p_values[fan_mask] = p_l * (a_fan / a_l) ** (2.0 * gamma / (gamma - 1.0))

        rho[left_mask] = rho_values
        u[left_mask] = u_values
        p[left_mask] = p_values

    right_mask = ~left_mask
    xi_right = xi[right_mask]

    if p_star > p_r:
        s_r = u_r + a_r * math.sqrt(((gamma + 1.0) / (2.0 * gamma)) * (p_star / p_r) + (gamma - 1.0) / (2.0 * gamma))
        right_state_mask = xi_right >= s_r
        rho[right_mask] = np.where(right_state_mask, rho_r, rho_star_r)
        u[right_mask] = np.where(right_state_mask, u_r, u_star)
        p[right_mask] = np.where(right_state_mask, p_r, p_star)
    else:
        s_head = u_r + a_r
        a_star_r = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
        s_tail = u_star + a_star_r
        right_state_mask = xi_right >= s_head
        star_mask = xi_right <= s_tail
        fan_mask = (~right_state_mask) & (~star_mask)

        rho_values = np.empty_like(xi_right)
        u_values = np.empty_like(xi_right)
        p_values = np.empty_like(xi_right)

        rho_values[right_state_mask] = rho_r
        u_values[right_state_mask] = u_r
        p_values[right_state_mask] = p_r

        rho_values[star_mask] = rho_star_r
        u_values[star_mask] = u_star
        p_values[star_mask] = p_star

        if np.any(fan_mask):
            xi_fan = xi_right[fan_mask]
            u_fan = (2.0 / (gamma + 1.0)) * (-a_r + 0.5 * (gamma - 1.0) * u_r + xi_fan)
            a_fan = (2.0 / (gamma + 1.0)) * (a_r - 0.5 * (gamma - 1.0) * (u_r - xi_fan))
            rho_values[fan_mask] = rho_r * (a_fan / a_r) ** (2.0 / (gamma - 1.0))
            u_values[fan_mask] = u_fan
            p_values[fan_mask] = p_r * (a_fan / a_r) ** (2.0 * gamma / (gamma - 1.0))

        rho[right_mask] = rho_values
        u[right_mask] = u_values
        p[right_mask] = p_values

    return rho, u, p


def build_exact_macro_rollout(
    case_params: Dict[str, float],
    steps: int,
    backend: str = "numpy",
    dtype: str = "float32",
    device=None,
):
    if steps <= 0:
        raise ValueError("steps must be positive.")

    gamma = float(case_params["vuy"])
    x = np.arange(int(case_params["X"]), dtype=np.float64)
    y_points = int(case_params["Y"])
    x0 = float(case_params.get("exact_x0", (int(case_params["X"]) // 2) + 0.5))
    left = dict(case_params["exact_left_state"])
    right = dict(case_params["exact_right_state"])
    numpy_dtype = resolve_numpy_dtype(dtype)

    rho_all = np.empty((steps, y_points, x.shape[0]), dtype=numpy_dtype)
    ux_all = np.empty_like(rho_all)
    uy_all = np.zeros_like(rho_all)
    temp_all = np.empty_like(rho_all)

    for step in range(steps):
        rho_1d, ux_1d, p_1d = solve_exact_riemann(x, float(step), left, right, gamma, x0)
        temp_1d = p_1d / rho_1d
        rho_all[step] = np.repeat(rho_1d[None, :], y_points, axis=0)
        ux_all[step] = np.repeat(ux_1d[None, :], y_points, axis=0)
        temp_all[step] = np.repeat(temp_1d[None, :], y_points, axis=0)

    backend_name = str(backend).lower()
    if backend_name == "numpy":
        return rho_all, ux_all, uy_all, temp_all
    if backend_name == "torch":
        torch_dtype = resolve_torch_dtype(dtype)
        return (
            torch.as_tensor(rho_all, dtype=torch_dtype, device=device),
            torch.as_tensor(ux_all, dtype=torch_dtype, device=device),
            torch.as_tensor(uy_all, dtype=torch_dtype, device=device),
            torch.as_tensor(temp_all, dtype=torch_dtype, device=device),
        )
    raise ValueError(f"Unsupported backend '{backend}'. Choose 'numpy' or 'torch'.")
