import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def default_config_path():
    return os.path.join(MODULE_DIR, "buckley_leverett_param_d2q9.yml")


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


def _stack_batch_results(results):
    first = results[0]
    if torch.is_tensor(first):
        return torch.stack(results, dim=0)
    if isinstance(first, tuple):
        return tuple(_stack_batch_results([result[idx] for result in results]) for idx in range(len(first)))
    raise TypeError(f"Unsupported batch result type: {type(first)!r}")


def initial_riemann(x, u_left, u_right, x0):
    return np.where(x <= x0, u_left, u_right).astype(np.float64)


def buckley_flux(u, mobility_ratio=0.5):
    denom = u**2 + mobility_ratio * (1.0 - u) ** 2 + 1e-12
    return u**2 / denom


def buckley_flux_prime(u, mobility_ratio=0.5):
    denom = u**2 + mobility_ratio * (1.0 - u) ** 2 + 1e-12
    return 2.0 * mobility_ratio * u * (1.0 - u) / (denom**2)


def reference_rollout(x, steps, dt, u_left, u_right, x0, mobility_ratio=0.5, reference_factor=8, cfl=0.45):
    if steps <= 0:
        return np.empty((0, len(x)), dtype=np.float32)

    x = np.asarray(x, dtype=np.float64)
    x_ref = np.linspace(x[0], x[-1], reference_factor * (len(x) - 1) + 1, dtype=np.float64)
    dx_ref = x_ref[1] - x_ref[0]
    u_ref = initial_riemann(x_ref, u_left, u_right, x0)
    rollout = []
    current_t = 0.0

    for step in range(steps):
        target_t = step * dt
        while current_t < target_t - 1e-12:
            speeds = np.abs(buckley_flux_prime(u_ref, mobility_ratio=mobility_ratio))
            max_speed = max(float(np.max(speeds)), 1e-6)
            dt_ref = min(cfl * dx_ref / max_speed, target_t - current_t)

            u_ext = np.concatenate([[u_left], u_ref, [u_right]])
            flux_vals = buckley_flux(u_ext, mobility_ratio=mobility_ratio)
            wave_speeds = np.maximum(
                np.abs(buckley_flux_prime(u_ext[:-1], mobility_ratio=mobility_ratio)),
                np.abs(buckley_flux_prime(u_ext[1:], mobility_ratio=mobility_ratio)),
            )
            num_flux = 0.5 * (flux_vals[:-1] + flux_vals[1:]) - 0.5 * wave_speeds * (u_ext[1:] - u_ext[:-1])
            u_ref = np.clip(u_ref - (dt_ref / dx_ref) * (num_flux[1:] - num_flux[:-1]), 0.0, 1.0)
            current_t += dt_ref

        rollout.append(u_ref[::reference_factor].astype(np.float32).copy())

    return np.stack(rollout, axis=0)


class BuckleyLeverettSolver(nn.Module):
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
        u_left_bc=None,
        u_right_bc=None,
        mobility_ratio=0.5,
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
        self.u_left_bc = u_left_bc
        self.u_right_bc = u_right_bc
        self.mobility_ratio = mobility_ratio
        self.dx = self.domain_length / max(self.X - 1, 1)
        self.dt = self.dx / self.lam
        self.directions, self.weights = self._build_lattice(self.lattice, device)
        self.velocities = self.lam * self.directions.to(dtype=torch.float32)
        self.Qn = int(self.velocities.numel())

    @staticmethod
    def _build_lattice(lattice, device):
        if lattice == "D1Q2":
            directions = torch.tensor([-1.0, 1.0], dtype=torch.float32, device=device)
            weights = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
            return directions, weights
        if lattice == "D1Q5":
            directions = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device=device)
            weights = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32, device=device) / 16.0
            return directions, weights
        if lattice == "D2Q9":
            directions = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0], dtype=torch.float32, device=device)
            weights = torch.tensor([1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 4.0 / 9.0], dtype=torch.float32, device=device)
            return directions, weights
        raise ValueError(f"Unsupported Buckley-Leverett lattice: {lattice}")

    def basis(self):
        return self.velocities.unsqueeze(-1)

    def x_grid(self):
        return np.linspace(0.0, self.domain_length, self.X, dtype=np.float64)

    def physical_x0(self, x0):
        return float(x0) if float(x0) <= self.domain_length else float(x0) * self.dx

    def flux(self, u):
        denom = u**2 + self.mobility_ratio * (1.0 - u) ** 2 + 1e-12
        return u**2 / denom

    def equilibrium_newton(self, u):
        if u.dim() > 1:
            return _stack_batch_results([self.equilibrium_newton(u[idx]) for idx in range(u.shape[0])])
        eps = 1e-8
        target_mass = u.clamp(0.0, 1.0)
        target_flux = self.flux(target_mass)
        equilibrium = torch.zeros((self.Qn, *u.shape), dtype=u.dtype, device=u.device)
        active = target_mass > eps
        if not torch.any(active):
            return equilibrium

        active_mass = target_mass[active]
        active_flux = target_flux[active]
        velocities = self.velocities.to(dtype=u.dtype, device=u.device)
        weights = self.weights.to(dtype=u.dtype, device=u.device)
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
        return equilibrium

    def equilibrium(self, u):
        if u.dim() > 1:
            return _stack_batch_results([self.equilibrium(u[idx]) for idx in range(u.shape[0])])
        if self.lattice == "D1Q2":
            flux = self.flux(u)
            f_minus = 0.5 * (u - flux / self.lam)
            f_plus = 0.5 * (u + flux / self.lam)
            return torch.stack([f_minus, f_plus], dim=0)
        return self.equilibrium_newton(u)

    def macro(self, F):
        return F.sum(dim=-2)

    def boundary_equilibrium(self, u_value, dtype):
        if u_value is None:
            return None
        u_tensor = torch.tensor([u_value], dtype=dtype, device=self.device)
        return self.equilibrium(u_tensor).squeeze(-1)

    def collision(self, F, Feq):
        return F - self.omega * (F - Feq)

    def streaming(self, F):
        if F.dim() > 2:
            return _stack_batch_results([self.streaming(F[idx]) for idx in range(F.shape[0])])
        streamed = torch.empty_like(F)
        left_eq = None
        right_eq = None
        if self.boundary == "riemann":
            left_eq = self.boundary_equilibrium(self.u_left_bc, F.dtype)
            right_eq = self.boundary_equilibrium(self.u_right_bc, F.dtype)
        for idx, direction in enumerate(self.directions.to(dtype=torch.int64).tolist()):
            if direction == 0:
                streamed[idx] = F[idx]
                continue
            shift = abs(int(direction))
            if self.boundary == "periodic":
                streamed[idx] = torch.roll(F[idx], shifts=direction, dims=0)
                continue
            if direction > 0:
                streamed[idx, shift:] = F[idx, :-shift]
                streamed[idx, :shift] = left_eq[idx] if self.boundary == "riemann" and left_eq is not None else F[idx, 0]
            else:
                streamed[idx, :-shift] = F[idx, shift:]
                streamed[idx, -shift:] = right_eq[idx] if self.boundary == "riemann" and right_eq is not None else F[idx, -1]
        return streamed

    def step(self, F, Feq=None):
        u = self.macro(F)
        if Feq is None:
            Feq = self.equilibrium(u)
        F_next = self.streaming(self.collision(F, Feq))
        return F_next, u, Feq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    data_path = resolve_module_path(config["data_dir"])
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    solver = BuckleyLeverettSolver(
        X=config["X"],
        lam=config["lam"],
        omega=config["omega"],
        lattice=config.get("lattice", "D2Q9"),
        device=args.device,
        newton_steps=config.get("newton_steps", 50),
        newton_tol=config.get("newton_tol", 1e-10),
        domain_length=config.get("domain_length", 1.0),
        boundary=config.get("boundary", "riemann"),
        u_left_bc=config.get("u_left"),
        u_right_bc=config.get("u_right"),
        mobility_ratio=config.get("mobility_ratio", 0.5),
    )

    x = solver.x_grid()
    x0 = solver.physical_x0(config["x0"])
    all_u = []
    all_Feq = []
    reference = reference_rollout(
        x=x,
        steps=args.steps,
        dt=solver.dt,
        u_left=float(config["u_left"]),
        u_right=float(config["u_right"]),
        x0=x0,
        mobility_ratio=config.get("mobility_ratio", 0.5),
        reference_factor=config.get("reference_factor", 8),
        cfl=config.get("reference_cfl", 0.45),
    )
    for step in range(args.steps):
        u = reference[step]
        all_u.append(u.astype(np.float32))
        feq = solver.equilibrium(torch.tensor(u, dtype=torch.float32, device=solver.device))
        all_Feq.append(feq.cpu().numpy().astype(np.float32))

    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("u", data=np.stack(all_u))
        handle.create_dataset("Feq", data=np.stack(all_Feq))
        handle.attrs["lattice"] = solver.lattice
        handle.attrs["Qn"] = solver.Qn
        handle.attrs["dt"] = solver.dt
        handle.attrs["dx"] = solver.dx
        handle.create_dataset("velocities", data=solver.velocities.cpu().numpy())

    print(f"Saved Buckley-Leverett dataset to {data_path}")


if __name__ == "__main__":
    main()
