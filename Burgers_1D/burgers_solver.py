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
    return os.path.join(MODULE_DIR, "burgers_param_d2q9.yml")


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


def default_scratch_module_dir():
    override = os.environ.get("NEURDE_BURGERS_ARTIFACT_ROOT")
    if override:
        return override

    normalized = os.path.normpath(MODULE_DIR)
    parts = normalized.split(os.sep)
    if len(parts) >= 4 and parts[1] == "home":
        return os.path.join(os.sep, "scratch", parts[2], *parts[3:])
    return MODULE_DIR


def resolve_artifact_root(artifact_root=None):
    if artifact_root in (None, "", False):
        return MODULE_DIR
    if os.path.isabs(str(artifact_root)):
        return str(artifact_root)

    selected_root = str(artifact_root).lower()
    if selected_root in {"module", "local", "workspace"}:
        return MODULE_DIR
    if selected_root == "scratch":
        return default_scratch_module_dir()
    raise ValueError(
        f"Unsupported Burgers artifact_root '{artifact_root}'. "
        "Use an absolute path or one of: module, local, workspace, scratch."
    )


def resolve_artifact_path(path, artifact_root=None):
    if os.path.isabs(path):
        return path
    return os.path.join(resolve_artifact_root(artifact_root), path)


def resolve_stabilizer_kwargs(config):
    return {
        "macro_limiter": config.get("macro_limiter", "none"),
        "macro_range_min": config.get("macro_range_min"),
        "macro_range_max": config.get("macro_range_max"),
        "macro_target_mean": config.get("macro_target_mean"),
    }


def get_model_config(config):
    model_config = dict(config.get("model", {}))
    if "feq_mode" not in model_config:
        conservative_output = config.get("conservative_output")
        if conservative_output is None:
            model_config["feq_mode"] = "positive"
        else:
            model_config["feq_mode"] = "projected_positive" if bool(conservative_output) else "positive"
    model_config["feq_mode"] = str(model_config["feq_mode"]).lower()
    model_config["logit_clip"] = float(model_config.get("logit_clip", config.get("logit_clip", 15.0)))
    return model_config


def exact_burgers_riemann(x, t, u_left, u_right, x0):
    if t <= 0:
        return np.where(x <= x0, u_left, u_right)

    xi = (x - x0) / t
    if u_left > u_right:
        shock_speed = 0.5 * (u_left + u_right)
        return np.where(xi <= shock_speed, u_left, u_right)

    result = np.empty_like(x, dtype=np.float64)
    result[xi <= u_left] = u_left
    result[xi >= u_right] = u_right
    fan_mask = (xi > u_left) & (xi < u_right)
    result[fan_mask] = xi[fan_mask]
    return result


def burgers_sinusoidal_initial(x, mean, amplitude, wavenumber=1.0, phase=0.0, domain_length=1.0):
    x = np.asarray(x, dtype=np.float64)
    phase_argument = (2.0 * np.pi * float(wavenumber) * x / float(domain_length)) + float(phase)
    return (float(mean) + float(amplitude) * np.sin(phase_argument)).astype(np.float64)


def burgers_sinusoidal_shock_time(amplitude, wavenumber=1.0, domain_length=1.0):
    slope_scale = 2.0 * np.pi * abs(float(amplitude)) * abs(float(wavenumber)) / float(domain_length)
    if slope_scale <= 1.0e-14:
        return np.inf
    return 1.0 / slope_scale


class BurgersSolver(nn.Module):
    def __init__(
        self,
        X=401,
        lam=1.0,
        omega=1.0,
        lattice="D1Q2",
        equilibrium_mode="default",
        device="cpu",
        newton_steps=50,
        newton_tol=1e-10,
        domain_length=1.0,
        alpha=0.5,
        s2=1.7,
        s3=1.7,
        boundary="outflow",
        u_left_bc=None,
        u_right_bc=None,
        macro_limiter="none",
        macro_range_min=None,
        macro_range_max=None,
        macro_target_mean=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.X = X
        self.lam = lam
        self.omega = omega
        self.lattice = lattice.upper()
        self.equilibrium_mode = equilibrium_mode.lower()
        self.device = device
        self.newton_steps = newton_steps
        self.newton_tol = newton_tol
        self.domain_length = float(domain_length)
        self.alpha = alpha
        self.s2 = s2
        self.s3 = s3
        self.boundary = boundary.lower()
        self.u_left_bc = u_left_bc
        self.u_right_bc = u_right_bc
        self.macro_limiter = str(macro_limiter).lower()
        self.macro_range_min = None if macro_range_min is None else float(macro_range_min)
        self.macro_range_max = None if macro_range_max is None else float(macro_range_max)
        self.macro_target_mean = None if macro_target_mean is None else float(macro_target_mean)
        self.dtype = resolve_torch_dtype(dtype)
        self.dx = self.domain_length / max(self.X - 1, 1)
        self.dt = self.dx / self.lam
        self.directions, self.weights = self._build_lattice(self.lattice, device, self.dtype)
        self.velocities = self.lam * self.directions.to(dtype=self.dtype)
        self.basis_vectors = self._build_basis(self.lattice, device, self.dtype)
        self.Qn = int(self.velocities.numel())
        if self.lattice == "D1Q3":
            self.M = torch.tensor(
                [
                    [1.0, 1.0, 1.0],
                    [-self.lam, 0.0, self.lam],
                    [self.lam**2, 0.0, self.lam**2],
                ],
                dtype=self.dtype,
                device=device,
            )
            self.M_inv = torch.inverse(self.M)
        if self.lattice == "D2Q9":
            self.d2q9_direction_groups = (
                (-1.0, torch.where(self.directions == -1.0)[0]),
                (0.0, torch.where(self.directions == 0.0)[0]),
                (1.0, torch.where(self.directions == 1.0)[0]),
            )

    @staticmethod
    def _build_lattice(lattice, device, dtype):
        if lattice == "D1Q2":
            directions = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)
            weights = torch.tensor([0.5, 0.5], dtype=dtype, device=device)
            return directions, weights
        if lattice == "D1Q3":
            directions = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype, device=device)
            weights = torch.tensor([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=dtype, device=device)
            return directions, weights
        if lattice == "D1Q5":
            directions = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype, device=device)
            weights = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=dtype, device=device) / 16.0
            return directions, weights
        if lattice == "D2Q9":
            directions = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0], dtype=dtype, device=device)
            weights = torch.tensor([1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 4.0 / 9.0], dtype=dtype, device=device)
            return directions, weights
        raise ValueError(f"Unsupported Burgers lattice: {lattice}")

    def _build_basis(self, lattice, device, dtype):
        if lattice == "D2Q9":
            # Use the full D2Q9 velocity basis so the network can distinguish
            # cardinal, diagonal, and rest populations that share the same x-velocity.
            ex = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0], dtype=dtype, device=device)
            ey = torch.tensor([0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0, 0.0], dtype=dtype, device=device)
            return self.lam * torch.stack([ex, ey], dim=-1)
        return self.velocities.unsqueeze(-1)

    def basis(self):
        return self.basis_vectors

    def x_grid(self):
        return np.linspace(0.0, self.domain_length, self.X, dtype=np.float64)

    def physical_x0(self, x0):
        return float(x0) if float(x0) <= self.domain_length else float(x0) * self.dx

    def equilibrium_newton(self, u):
        eps = 1e-8
        leading_shape = u.shape[:-1]
        target_mass = u.clamp_min(0.0).reshape(-1)
        target_flux = 0.5 * target_mass**2
        equilibrium = torch.zeros((self.Qn, target_mass.numel()), dtype=u.dtype, device=u.device)
        active = target_mass > eps
        if not torch.any(active):
            return equilibrium.transpose(0, 1).reshape(*leading_shape, u.shape[-1], self.Qn).movedim(-1, -2)

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
            delta_alpha = (res0 * j11 - res1 * j01) / det
            delta_beta = (j00 * res1 - j01 * res0) / det
            alpha = alpha - delta_alpha
            beta = beta - delta_beta

        exponent = logw + alpha[None, :] + beta[None, :] * vel
        equilibrium[:, active] = torch.exp(exponent)
        return equilibrium.transpose(0, 1).reshape(*leading_shape, u.shape[-1], self.Qn).movedim(-1, -2)

    def equilibrium(self, u):
        u = self.stabilize_macro(u)
        if self.lattice == "D1Q3":
            return self.equilibrium_d1q3(u)
        if self.lattice == "D2Q9":
            return self.equilibrium_d2q9(u)
        if self.lattice == "D1Q2":
            flux = 0.5 * u**2
            f_minus = 0.5 * (u - flux / self.lam)
            f_plus = 0.5 * (u + flux / self.lam)
            return torch.stack([f_minus, f_plus], dim=-2)
        return self.equilibrium_newton(u)

    def equilibrium_d1q3_with_mode(self, u, mode):
        if mode in ("default", "centered"):
            f_plus = 0.5 * self.alpha * u + (u**2) / (4.0 * self.lam)
            f_zero = (1.0 - self.alpha) * u
            f_minus = 0.5 * self.alpha * u - (u**2) / (4.0 * self.lam)
            return torch.stack([f_minus, f_zero, f_plus], dim=-2)
        if mode == "upwind":
            nonnegative = u >= 0
            f_plus = torch.where(nonnegative, (u**2) / (2.0 * self.lam), torch.zeros_like(u))
            f_zero = torch.where(nonnegative, u - (u**2) / (2.0 * self.lam), u + (u**2) / (2.0 * self.lam))
            f_minus = torch.where(nonnegative, torch.zeros_like(u), -(u**2) / (2.0 * self.lam))
            return torch.stack([f_minus, f_zero, f_plus], dim=-2)
        raise ValueError(f"Unsupported D1Q3 equilibrium mode: {mode}")

    def equilibrium_d1q3(self, u):
        return self.equilibrium_d1q3_with_mode(u, self.equilibrium_mode)

    def equilibrium_d2q9(self, u):
        selected_mode = "upwind" if self.equilibrium_mode == "default" else self.equilibrium_mode
        if selected_mode in ("centered", "upwind"):
            reduced_equilibrium = self.equilibrium_d1q3_with_mode(u, selected_mode)
            equilibrium = torch.zeros((*u.shape[:-1], self.Qn, u.shape[-1]), dtype=u.dtype, device=u.device)
            reduced_by_direction = {
                -1.0: reduced_equilibrium[..., 0, :],
                0.0: reduced_equilibrium[..., 1, :],
                1.0: reduced_equilibrium[..., 2, :],
            }
            for direction, indices in self.d2q9_direction_groups:
                group_weights = self.weights[indices].to(dtype=u.dtype, device=u.device)
                group_total = group_weights.sum()
                view_shape = (1,) * len(u.shape[:-1]) + (group_weights.shape[0], 1)
                equilibrium[..., indices, :] = (
                    group_weights.view(view_shape) / group_total
                ) * reduced_by_direction[direction].unsqueeze(-2)
            return equilibrium
        if selected_mode in ("newton", "maxent"):
            return self.equilibrium_newton(u)
        raise ValueError(f"Unsupported D2Q9 equilibrium mode: {self.equilibrium_mode}")

    def equilibrium_moment_d1q3(self, u):
        m1 = u
        m2 = 0.5 * u**2
        if self.equilibrium_mode in ("default", "centered"):
            m3 = self.alpha * (self.lam**2) * u
        elif self.equilibrium_mode == "upwind":
            m3 = self.lam * torch.sign(u) * 0.5 * u**2
        else:
            raise ValueError(f"Unsupported D1Q3 equilibrium mode: {self.equilibrium_mode}")
        return torch.stack([m1, m2, m3], dim=-2)

    def macro(self, F):
        return F.sum(dim=-2)

    def boundary_equilibrium(self, u_value, dtype):
        if u_value is None:
            return None
        u_tensor = torch.tensor([u_value], dtype=dtype, device=self.device)
        return self.equilibrium(u_tensor).squeeze(-1)

    def collision(self, F, Feq):
        if self.lattice == "D1Q3":
            return self.collision_d1q3(F, Feq)
        return F - self.omega * (F - Feq)

    def collision_d1q3(self, F, Feq=None):
        if Feq is None:
            Feq = self.equilibrium(self.macro(F))
        moments = torch.einsum("ab,...bx->...ax", self.M, F)
        moments_eq = torch.einsum("ab,...bx->...ax", self.M, Feq)
        moments_star = moments.clone()
        moments_star[..., 0, :] = moments_eq[..., 0, :]
        moments_star[..., 1, :] = moments[..., 1, :] + self.s2 * (moments_eq[..., 1, :] - moments[..., 1, :])
        moments_star[..., 2, :] = moments[..., 2, :] + self.s3 * (moments_eq[..., 2, :] - moments[..., 2, :])
        return torch.einsum("ab,...bx->...ax", self.M_inv, moments_star)

    def streaming(self, F):
        streamed = torch.empty_like(F)
        left_eq = None
        right_eq = None
        if self.boundary == "riemann":
            left_eq = self.boundary_equilibrium(self.u_left_bc, F.dtype)
            right_eq = self.boundary_equilibrium(self.u_right_bc, F.dtype)
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
        u = self.macro(F)
        if Feq is None:
            Feq = self.equilibrium(u)
        F_next = self.streaming(self.collision(F, Feq))
        F_next = self.stabilize_population(F_next)
        return F_next, u, Feq

    def _project_macro_box_preserve_mean(self, u):
        lower = self.macro_range_min if self.macro_range_min is not None else None
        upper = self.macro_range_max if self.macro_range_max is not None else None
        if self.macro_limiter == "none":
            return u

        replacement = self.macro_target_mean if self.macro_target_mean is not None else 0.0
        if lower is not None:
            replacement = max(replacement, lower)
        if upper is not None:
            replacement = min(replacement, upper)
        limited = torch.nan_to_num(u, nan=replacement, posinf=replacement, neginf=replacement)

        if self.macro_limiter == "clamp":
            if lower is not None or upper is not None:
                min_value = lower if lower is not None else -torch.inf
                max_value = upper if upper is not None else torch.inf
                limited = limited.clamp(min=min_value, max=max_value)
            return limited

        if self.macro_limiter != "conservative_clamp":
            raise ValueError(f"Unsupported macro_limiter: {self.macro_limiter}")

        if lower is None or upper is None:
            raise ValueError("conservative_clamp requires both macro_range_min and macro_range_max.")

        target_mean = self.macro_target_mean if self.macro_target_mean is not None else float(limited.mean())
        lower_bound = lower - float(torch.max(limited))
        upper_bound = upper - float(torch.min(limited))
        for _ in range(64):
            offset = 0.5 * (lower_bound + upper_bound)
            candidate = torch.clamp(limited + offset, min=lower, max=upper)
            if float(candidate.mean()) < target_mean:
                lower_bound = offset
            else:
                upper_bound = offset
        return torch.clamp(limited + 0.5 * (lower_bound + upper_bound), min=lower, max=upper)

    def stabilize_macro(self, u):
        if self.macro_limiter == "none":
            return u

        has_bad_macro = not torch.isfinite(u).all()
        if self.macro_range_min is not None:
            has_bad_macro = has_bad_macro or bool((u < self.macro_range_min).any())
        if self.macro_range_max is not None:
            has_bad_macro = has_bad_macro or bool((u > self.macro_range_max).any())
        if not has_bad_macro:
            return u
        return self._project_macro_box_preserve_mean(u)

    def stabilize_population(self, F):
        if self.macro_limiter == "none":
            return F

        u = self.macro(F)
        limited_u = self.stabilize_macro(u)
        has_bad_population = not torch.isfinite(F).all()
        has_bad_macro = not torch.equal(limited_u, u)
        if not has_bad_population and not has_bad_macro:
            return F

        return self.equilibrium(limited_u)


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
    artifact_root = config.get("artifact_root")
    data_path = resolve_artifact_path(config["data_dir"], artifact_root=artifact_root)

    torch_dtype = resolve_torch_dtype(args.dtype)
    numpy_dtype = resolve_numpy_dtype(args.dtype)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    solver = BurgersSolver(
        X=config["X"],
        lam=config["lam"],
        omega=config["omega"],
        lattice=config.get("lattice", "D1Q2"),
        equilibrium_mode=config.get("equilibrium_mode", "default"),
        device=args.device,
        newton_steps=config.get("newton_steps", 50),
        newton_tol=config.get("newton_tol", 1e-10),
        domain_length=config.get("domain_length", 1.0),
        alpha=config.get("alpha", 0.5),
        s2=config.get("s2", 1.7),
        s3=config.get("s3", 1.7),
        boundary=config.get("boundary", "outflow"),
        u_left_bc=config.get("u_left"),
        u_right_bc=config.get("u_right"),
        dtype=torch_dtype,
        **resolve_stabilizer_kwargs(config),
    )

    x = solver.x_grid()
    all_u = []
    all_F = []
    all_Feq = []
    initial_condition = str(config.get("initial_condition", "riemann")).lower()
    shock_time = None

    if initial_condition == "sinusoidal":
        u0 = burgers_sinusoidal_initial(
            x,
            mean=config.get("sine_mean", 0.6),
            amplitude=config.get("sine_amplitude", 0.4),
            wavenumber=config.get("sine_wavenumber", 1.0),
            phase=config.get("sine_phase", 0.0),
            domain_length=config.get("domain_length", 1.0),
        )
        shock_time = burgers_sinusoidal_shock_time(
            amplitude=config.get("sine_amplitude", 0.4),
            wavenumber=config.get("sine_wavenumber", 1.0),
            domain_length=config.get("domain_length", 1.0),
        )
        F = solver.equilibrium(torch.tensor(u0, dtype=torch_dtype, device=solver.device))
    else:
        x0 = solver.physical_x0(config["x0"])
        u0 = exact_burgers_riemann(x, 0.0, config["u_left"], config["u_right"], x0)
        F = solver.equilibrium(torch.tensor(u0, dtype=torch_dtype, device=solver.device))

    with torch.no_grad():
        for step in range(args.steps):
            u_tensor = solver.macro(F)
            feq = solver.equilibrium(u_tensor)
            all_u.append(u_tensor.detach().cpu().numpy().astype(numpy_dtype))
            all_F.append(F.detach().cpu().numpy().astype(numpy_dtype))
            all_Feq.append(feq.detach().cpu().numpy().astype(numpy_dtype))
            if step < args.steps - 1:
                F, _, _ = solver.step(F, feq)

    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("u", data=np.stack(all_u))
        handle.create_dataset("F", data=np.stack(all_F))
        handle.create_dataset("Feq", data=np.stack(all_Feq))
        handle.attrs["lattice"] = solver.lattice
        handle.attrs["equilibrium_mode"] = solver.equilibrium_mode
        handle.attrs["initial_condition"] = initial_condition
        handle.attrs["Qn"] = solver.Qn
        handle.attrs["dt"] = solver.dt
        handle.attrs["dx"] = solver.dx
        handle.create_dataset("velocities", data=solver.velocities.cpu().numpy())
        handle.create_dataset("x", data=x.astype(numpy_dtype))
        if shock_time is not None and np.isfinite(shock_time):
            handle.attrs["estimated_shock_time"] = float(shock_time)

    print(f"Saved Burgers dataset to {data_path}")
    if shock_time is not None and np.isfinite(shock_time):
        print(f"Estimated sinusoidal shock time: {shock_time:.6f} (step ~ {shock_time / solver.dt:.1f})")


if __name__ == "__main__":
    main()
