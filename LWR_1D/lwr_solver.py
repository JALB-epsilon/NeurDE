import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}
_NUMPY_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
}


def resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        if dtype not in _TORCH_DTYPE_MAP.values():
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        return dtype
    dtype_name = str(dtype).lower()
    if dtype_name not in _TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {sorted(_TORCH_DTYPE_MAP)}")
    return _TORCH_DTYPE_MAP[dtype_name]


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


def default_config_path():
    return os.path.join(MODULE_DIR, "lwr_param_d2q9.yml")


def resolve_config_path(path):
    if os.path.isabs(path):
        return path
    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(MODULE_DIR, path)


def resolve_module_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(MODULE_DIR, path)


def initial_riemann(x, rho_left, rho_right, x0):
    return np.where(x <= x0, rho_left, rho_right).astype(np.float64)


def lwr_flux(rho, rho_max=1.0, v_free=1.0):
    return v_free * rho * (1.0 - rho / rho_max)


def lwr_flux_prime(rho, rho_max=1.0, v_free=1.0):
    return v_free * (1.0 - 2.0 * rho / rho_max)


def exact_lwr_greenshields_riemann(x, t, rho_left, rho_right, x0, rho_max=1.0, v_free=1.0):
    if t <= 0:
        return initial_riemann(x, rho_left, rho_right, x0)

    xi = (x - x0) / t
    left_speed = lwr_flux_prime(rho_left, rho_max=rho_max, v_free=v_free)
    right_speed = lwr_flux_prime(rho_right, rho_max=rho_max, v_free=v_free)
    flux_left = lwr_flux(rho_left, rho_max=rho_max, v_free=v_free)
    flux_right = lwr_flux(rho_right, rho_max=rho_max, v_free=v_free)

    if rho_left < rho_right:
        shock_speed = (flux_right - flux_left) / (rho_right - rho_left + 1e-12)
        return np.where(xi <= shock_speed, rho_left, rho_right)

    result = np.empty_like(x, dtype=np.float64)
    result[xi <= left_speed] = rho_left
    result[xi >= right_speed] = rho_right
    fan_mask = (xi > left_speed) & (xi < right_speed)
    result[fan_mask] = 0.5 * rho_max * (1.0 - xi[fan_mask] / v_free)
    return np.clip(result, 0.0, rho_max)


class LWRSolver(nn.Module):
    def __init__(
        self,
        X=401,
        lam=1.0,
        omega=1.0,
        lattice="D2Q9",
        device="cpu",
        newton_steps=50,
        newton_tol=1e-10,
        domain_length=1.0,
        boundary="riemann",
        rho_left_bc=None,
        rho_right_bc=None,
        rho_max=1.0,
        v_free=1.0,
        dtype=torch.float32,
    ):
        super().__init__()
        self.X = X
        self.lam = lam
        self.omega = omega
        self.lattice = lattice.upper()
        self.device = device
        self.newton_steps = newton_steps
        self.newton_tol = newton_tol
        self.domain_length = float(domain_length)
        self.boundary = boundary.lower()
        self.rho_left_bc = rho_left_bc
        self.rho_right_bc = rho_right_bc
        self.rho_max = rho_max
        self.v_free = v_free
        self.dtype = resolve_torch_dtype(dtype)
        self.dx = self.domain_length / max(self.X - 1, 1)
        self.dt = self.dx / self.lam
        self.directions, self.weights = self._build_lattice(self.lattice, device, self.dtype)
        self.velocities = self.lam * self.directions.to(dtype=self.dtype)
        self.Qn = int(self.velocities.numel())

    @staticmethod
    def _build_lattice(lattice, device, dtype):
        if lattice == "D1Q2":
            directions = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)
            weights = torch.tensor([0.5, 0.5], dtype=dtype, device=device)
            return directions, weights
        if lattice == "D1Q5":
            directions = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype, device=device)
            weights = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=dtype, device=device) / 16.0
            return directions, weights
        if lattice == "D2Q9":
            directions = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0], dtype=dtype, device=device)
            weights = torch.tensor([1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 4.0 / 9.0], dtype=dtype, device=device)
            return directions, weights
        raise ValueError(f"Unsupported LWR lattice: {lattice}")

    def basis(self):
        return self.velocities.unsqueeze(-1)

    def x_grid(self):
        return np.linspace(0.0, self.domain_length, self.X, dtype=np.float64)

    def physical_x0(self, x0):
        return float(x0) if float(x0) <= self.domain_length else float(x0) * self.dx

    def flux(self, rho):
        return self.v_free * rho * (1.0 - rho / self.rho_max)

    def equilibrium_newton(self, rho):
        eps = 1e-8
        leading_shape = rho.shape[:-1]
        target_mass = rho.clamp(0.0, self.rho_max).reshape(-1)
        target_flux = self.flux(target_mass)
        equilibrium = torch.zeros((self.Qn, target_mass.numel()), dtype=rho.dtype, device=rho.device)
        active = target_mass > eps
        if not torch.any(active):
            return equilibrium.transpose(0, 1).reshape(*leading_shape, rho.shape[-1], self.Qn).movedim(-1, -2)

        active_mass = target_mass[active]
        active_flux = target_flux[active]
        velocities = self.velocities.to(dtype=rho.dtype, device=rho.device)
        weights = self.weights.to(dtype=rho.dtype, device=rho.device)
        alpha = torch.log(active_mass)
        beta = torch.zeros_like(active_mass)
        vel = velocities[:, None]
        logw = torch.log(weights)[:, None]
        vel_sq = vel * vel

        for _ in range(self.newton_steps):
            exponent = logw + alpha[None, :] + beta[None, :] * vel
            f = torch.exp(exponent)
            mass = f.sum(dim=0)
            flux = (vel * f).sum(dim=0)
            res0 = mass - active_mass
            res1 = flux - active_flux
            if torch.max(torch.abs(res0)).item() < self.newton_tol and torch.max(torch.abs(res1)).item() < self.newton_tol:
                break

            j00 = mass
            j01 = flux
            j11 = (vel_sq * f).sum(dim=0)
            det = j00 * j11 - j01 * j01 + eps
            alpha = alpha - (res0 * j11 - res1 * j01) / det
            beta = beta - (j00 * res1 - j01 * res0) / det

        exponent = logw + alpha[None, :] + beta[None, :] * vel
        equilibrium[:, active] = torch.exp(exponent)
        return equilibrium.transpose(0, 1).reshape(*leading_shape, rho.shape[-1], self.Qn).movedim(-1, -2)

    def equilibrium(self, rho):
        if self.lattice == "D1Q2":
            flux = self.flux(rho)
            f_minus = 0.5 * (rho - flux / self.lam)
            f_plus = 0.5 * (rho + flux / self.lam)
            return torch.stack([f_minus, f_plus], dim=-2)
        return self.equilibrium_newton(rho)

    def macro(self, F):
        return F.sum(dim=-2)

    def boundary_equilibrium(self, rho_value, dtype):
        if rho_value is None:
            return None
        rho_tensor = torch.tensor([rho_value], dtype=dtype, device=self.device)
        return self.equilibrium(rho_tensor).squeeze(-1)

    def collision(self, F, Feq):
        return F - self.omega * (F - Feq)

    def streaming(self, F):
        streamed = torch.empty_like(F)
        left_eq = None
        right_eq = None
        if self.boundary == "riemann":
            left_eq = self.boundary_equilibrium(self.rho_left_bc, F.dtype)
            right_eq = self.boundary_equilibrium(self.rho_right_bc, F.dtype)
        for idx, direction in enumerate(self.directions.to(dtype=torch.int64).tolist()):
            if direction == 0:
                streamed[..., idx, :] = F[..., idx, :]
                continue
            shift = abs(int(direction))
            if self.boundary == "periodic":
                streamed[..., idx, :] = torch.roll(F[..., idx, :], shifts=direction, dims=-1)
                continue
            if direction > 0:
                streamed[..., idx, shift:] = F[..., idx, :-shift]
                if self.boundary == "riemann" and left_eq is not None:
                    streamed[..., idx, :shift] = left_eq[idx]
                else:
                    streamed[..., idx, :shift] = F[..., idx, 0].unsqueeze(-1)
            else:
                streamed[..., idx, :-shift] = F[..., idx, shift:]
                if self.boundary == "riemann" and right_eq is not None:
                    streamed[..., idx, -shift:] = right_eq[idx]
                else:
                    streamed[..., idx, -shift:] = F[..., idx, -1].unsqueeze(-1)
        return streamed

    def step(self, F, Feq=None):
        rho = self.macro(F)
        if Feq is None:
            Feq = self.equilibrium(rho)
        F_next = self.streaming(self.collision(F, Feq))
        return F_next, rho, Feq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    data_path = resolve_module_path(config["data_dir"])
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    torch_dtype = resolve_torch_dtype(args.dtype)
    numpy_dtype = resolve_numpy_dtype(args.dtype)
    solver = LWRSolver(
        X=config["X"],
        lam=config["lam"],
        omega=config["omega"],
        lattice=config.get("lattice", "D2Q9"),
        device=args.device,
        newton_steps=config.get("newton_steps", 50),
        newton_tol=config.get("newton_tol", 1e-10),
        domain_length=config.get("domain_length", 1.0),
        boundary=config.get("boundary", "riemann"),
        rho_left_bc=config.get("rho_left"),
        rho_right_bc=config.get("rho_right"),
        rho_max=config.get("rho_max", 1.0),
        v_free=config.get("v_free", 1.0),
        dtype=torch_dtype,
    )

    x = solver.x_grid()
    x0 = solver.physical_x0(config["x0"])
    all_rho = []
    all_Feq = []
    for step in range(args.steps):
        rho = exact_lwr_greenshields_riemann(
            x,
            step * solver.dt,
            config["rho_left"],
            config["rho_right"],
            x0,
            rho_max=config.get("rho_max", 1.0),
            v_free=config.get("v_free", 1.0),
        )
        all_rho.append(rho.astype(numpy_dtype))
        feq = solver.equilibrium(torch.tensor(rho, dtype=torch_dtype, device=solver.device))
        all_Feq.append(feq.cpu().numpy().astype(numpy_dtype))

    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("u", data=np.stack(all_rho))
        handle.create_dataset("Feq", data=np.stack(all_Feq))
        handle.attrs["lattice"] = solver.lattice
        handle.attrs["Qn"] = solver.Qn
        handle.attrs["dt"] = solver.dt
        handle.attrs["dx"] = solver.dx
        handle.create_dataset("velocities", data=solver.velocities.cpu().numpy())

    print(f"Saved LWR dataset to {data_path}")


if __name__ == "__main__":
    main()
