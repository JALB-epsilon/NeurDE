import argparse
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.entropic_lattice import (
    build_d2q49_lattice,
    compute_entropic_equilibrium,
    compute_geq_from_feq,
    estimate_entropic_alpha,
)
from utilities import detach, get_device, resolve_torch_dtype


def _shock_post_state(gamma, rho_pre, p_pre, shock_mach):
    gamma = float(gamma)
    rho_pre = float(rho_pre)
    p_pre = float(p_pre)
    shock_mach = float(shock_mach)
    temp_pre = p_pre / rho_pre
    sound_speed_pre = np.sqrt(gamma * temp_pre)
    shock_speed = shock_mach * sound_speed_pre
    density_ratio = ((gamma + 1.0) * shock_mach**2) / ((gamma - 1.0) * shock_mach**2 + 2.0)
    pressure_ratio = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (shock_mach**2 - 1.0)
    rho_post = rho_pre * density_ratio
    p_post = p_pre * pressure_ratio
    temp_post = p_post / rho_post
    ux_post = shock_speed * (1.0 - 1.0 / density_ratio)
    return {
        "rho": rho_post,
        "ux": ux_post,
        "uy": 0.0,
        "p": p_post,
        "T": temp_post,
        "shock_speed": shock_speed,
    }


def _triangle_polygon(apex_x, center_y, side_length, apex_points_right=True):
    half_side = 0.5 * side_length
    apex_dx = np.sqrt(3.0) * half_side
    if apex_points_right:
        face_center_x = apex_x - apex_dx
    else:
        face_center_x = apex_x + apex_dx
    return np.array(
        [
            [face_center_x, center_y + half_side],
            [face_center_x, center_y - half_side],
            [apex_x, center_y],
        ],
        dtype=np.float64,
    )


def _scaled_positive_int(value, scale, minimum=8):
    return max(int(round(float(value) * float(scale))), int(minimum))


class SchardinELBMSolver(nn.Module):
    def __init__(
        self,
        X,
        Y,
        wedge_side,
        apex_x,
        center_y,
        shock_x,
        shock_half_width=3.0,
        shock_mach=1.34,
        Re=2000.0,
        rho_pre=1.4,
        p_pre=1.0,
        vuy=1.4,
        Pr=0.71,
        shift_fraction=0.6,
        wall_rho=None,
        wall_T=None,
        wall_mode="no_slip",
        wall_thermal="adiabatic",
        apex_points_right=True,
        newton_iters=8,
        newton_tolerance=1e-8,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        self.X = int(X)
        self.Y = int(Y)
        self.wedge_side = int(wedge_side)
        self.apex_x = float(apex_x)
        self.center_y = float(center_y)
        self.shock_x = float(shock_x)
        self.shock_half_width = float(shock_half_width)
        self.shock_mach = float(shock_mach)
        self.Re = float(Re)
        self.rho_pre = float(rho_pre)
        self.p_pre = float(p_pre)
        self.vuy = float(vuy)
        self.Pr = float(Pr)
        self.shift_fraction = float(shift_fraction)
        self.apex_points_right = bool(apex_points_right)
        self.wall_mode = str(wall_mode).lower()
        self.wall_thermal = str(wall_thermal).lower()
        self.newton_iters = int(newton_iters)
        self.newton_tolerance = float(newton_tolerance)
        self.device = device
        self.dtype = resolve_torch_dtype(dtype)
        self.dimensions = 2.0
        self.pop_floor = 1e-12
        self.rho_floor = 1e-6
        self.temp_floor = 1e-6

        self.Cv = 1.0 / (self.vuy - 1.0)
        self.Cp = self.vuy * self.Cv
        self.R = self.Cp - self.Cv
        self.internal_factor = max(2.0 * self.Cv - self.dimensions, 0.0)

        self.pre_state = {
            "rho": self.rho_pre,
            "ux": 0.0,
            "uy": 0.0,
            "p": self.p_pre,
            "T": self.p_pre / self.rho_pre,
        }
        self.post_state = _shock_post_state(self.vuy, self.rho_pre, self.p_pre, self.shock_mach)
        self.wall_rho = float(self.pre_state["rho"] if wall_rho is None else wall_rho)
        self.wall_T = float(self.pre_state["T"] if wall_T is None else wall_T)

        self.U0 = float(self.post_state["ux"])
        self.Uax = self.U0 * self.shift_fraction
        self.Uay = 0.0
        if abs(self.Uax) >= 1.0 or abs(self.Uay) >= 1.0:
            raise ValueError("This implementation supports only fractional shifts with magnitude < 1.")

        self.lattice = build_d2q49_lattice(self.device, self.dtype, shift_x=self.Uax, shift_y=self.Uay)
        self.Qn = self.lattice.q
        self.ex = self.lattice.cx
        self.ey = self.lattice.cy
        self.ex_base = self.lattice.cx_base
        self.ey_base = self.lattice.cy_base
        self.speed_sq = self.lattice.speed_sq
        self.weights = self.lattice.weights
        self.opp = self.lattice.opp.to(device=self.device)

        vel_scale = max(abs(self.post_state["ux"]) + 4.0 * np.sqrt(self.vuy * self.post_state["T"]), 6.0)
        self.velocity_cap = float(vel_scale)
        self.rho_cap = max(20.0 * self.post_state["rho"], 20.0)
        self.temp_cap = max(20.0 * self.post_state["T"], 20.0)

        rho_ref = max(self.post_state["rho"], 1e-8)
        u_ref = max(abs(self.post_state["ux"]), 1e-8)
        self.muy = rho_ref * u_ref * self.wedge_side / self.Re

        self._create_streaming_indices()
        self._create_obstacle()
        self._initialize_reference_boundaries()

    def _create_streaming_indices(self):
        self.shifts_y = -self.ey_base.to(torch.long)
        self.shifts_x = self.ex_base.to(torch.long)
        self.q_indices = torch.arange(self.Qn, device=self.device)[:, None, None]
        y_indices = (torch.arange(self.Y, device=self.device)[None, :, None] - self.shifts_y[:, None, None]) % self.Y
        x_indices = (torch.arange(self.X, device=self.device)[None, None, :] - self.shifts_x[:, None, None]) % self.X
        self.Y_indices = y_indices.expand(self.Qn, self.Y, self.X)
        self.X_indices = x_indices.expand(self.Qn, self.Y, self.X)

    def _create_obstacle(self):
        x_coords = np.arange(self.X, dtype=np.float64) + 0.5
        y_coords = np.arange(self.Y - 1, -1, -1, dtype=np.float64) + 0.5
        xx, yy = np.meshgrid(x_coords, y_coords)
        polygon = _triangle_polygon(
            apex_x=self.apex_x,
            center_y=self.center_y,
            side_length=self.wedge_side,
            apex_points_right=self.apex_points_right,
        )
        mask = Path(polygon).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(self.Y, self.X)
        obs = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        obs[:, torch.sum(obs, dim=0) < 2] = 0
        obs[torch.sum(obs, dim=1) < 2, :] = 0
        self.Obs = obs
        self.Fluid = ~obs
        self.wall_missing_mask = self.Fluid.unsqueeze(0) & self.Obs[self.Y_indices, self.X_indices]

    def _initialize_reference_boundaries(self):
        inlet_rho = torch.full((self.Y, 1), self.post_state["rho"], device=self.device, dtype=self.dtype)
        inlet_ux = torch.full((self.Y, 1), self.post_state["ux"], device=self.device, dtype=self.dtype)
        inlet_uy = torch.zeros((self.Y, 1), device=self.device, dtype=self.dtype)
        inlet_T = torch.full((self.Y, 1), self.post_state["T"], device=self.device, dtype=self.dtype)
        self.inlet_lambdas = None
        inlet_feq, self.inlet_lambdas = self.compute_equilibrium(inlet_rho, inlet_ux, inlet_uy, inlet_T)
        inlet_geq = compute_geq_from_feq(inlet_feq, inlet_T, self.Cv, dimensions=self.dimensions)
        self.inlet_Feq = inlet_feq.squeeze(-1)
        self.inlet_Geq = inlet_geq.squeeze(-1)

        wall_rho = torch.full((1,), self.wall_rho, device=self.device, dtype=self.dtype)
        wall_ux = torch.zeros((1,), device=self.device, dtype=self.dtype)
        wall_uy = torch.zeros((1,), device=self.device, dtype=self.dtype)
        wall_T = torch.full((1,), self.wall_T, device=self.device, dtype=self.dtype)
        wall_feq, _ = self.compute_equilibrium(wall_rho, wall_ux, wall_uy, wall_T)
        wall_geq = compute_geq_from_feq(wall_feq, wall_T, self.Cv, dimensions=self.dimensions)
        self.wall_Feq_vector = wall_feq[:, 0]
        self.wall_Geq_vector = wall_geq[:, 0]

    def compute_equilibrium(self, rho, ux, uy, T, lambdas=None):
        return compute_entropic_equilibrium(
            rho,
            ux,
            uy,
            T,
            self.lattice,
            lambdas=lambdas,
            newton_iters=self.newton_iters,
            tolerance=self.newton_tolerance,
        )

    def _shock_profile(self):
        x = torch.arange(self.X, device=self.device, dtype=self.dtype)
        if self.shock_half_width <= 0.0:
            return (x <= self.shock_x).to(self.dtype)
        return 0.5 * (1.0 - torch.tanh((x - self.shock_x) / max(self.shock_half_width, 1e-6)))

    def initial_conditions(self):
        profile = self._shock_profile()[None, :].expand(self.Y, self.X)
        rho = self.pre_state["rho"] + profile * (self.post_state["rho"] - self.pre_state["rho"])
        ux = self.pre_state["ux"] + profile * (self.post_state["ux"] - self.pre_state["ux"])
        uy = torch.zeros((self.Y, self.X), device=self.device, dtype=self.dtype)
        T = self.pre_state["T"] + profile * (self.post_state["T"] - self.pre_state["T"])

        rho = torch.where(self.Obs, torch.full_like(rho, self.wall_rho), rho)
        ux = torch.where(self.Obs, torch.zeros_like(ux), ux)
        uy = torch.where(self.Obs, torch.zeros_like(uy), uy)
        T = torch.where(self.Obs, torch.full_like(T, self.wall_T), T)

        feq, lambdas = self.compute_equilibrium(rho, ux, uy, T)
        geq = compute_geq_from_feq(feq, T, self.Cv, dimensions=self.dimensions)
        feq[:, self.Obs] = self.wall_Feq_vector[:, None]
        geq[:, self.Obs] = self.wall_Geq_vector[:, None]
        return feq, geq, lambdas

    def get_macroscopic(self, F, G):
        rho = torch.sum(F, dim=0)
        inv_rho = 1.0 / rho.clamp_min(self.rho_floor)
        rho_ux = torch.tensordot(self.ex, F, dims=([0], [0]))
        rho_uy = torch.tensordot(self.ey, F, dims=([0], [0]))
        ux = rho_ux * inv_rho
        uy = rho_uy * inv_rho
        kinetic = torch.tensordot(self.speed_sq, F, dims=([0], [0]))
        internal = torch.sum(G, dim=0)
        T = (kinetic + internal - rho * (ux * ux + uy * uy)) / (2.0 * self.Cv * rho.clamp_min(self.rho_floor))
        return rho, ux, uy, T

    def sanitize_macroscopic(self, rho, ux, uy, T):
        rho = torch.nan_to_num(
            rho,
            nan=self.pre_state["rho"],
            posinf=self.rho_cap,
            neginf=self.rho_floor,
        ).clamp(min=self.rho_floor, max=self.rho_cap)
        ux = torch.nan_to_num(ux, nan=0.0, posinf=self.velocity_cap, neginf=-self.velocity_cap).clamp(
            min=-self.velocity_cap, max=self.velocity_cap
        )
        uy = torch.nan_to_num(uy, nan=0.0, posinf=self.velocity_cap, neginf=-self.velocity_cap).clamp(
            min=-self.velocity_cap, max=self.velocity_cap
        )
        T = torch.nan_to_num(
            T,
            nan=self.pre_state["T"],
            posinf=self.temp_cap,
            neginf=self.temp_floor,
        ).clamp(min=self.temp_floor, max=self.temp_cap)
        return rho, ux, uy, T

    def get_relaxation_parameters(self, rho, T):
        tau_mom = self.muy / (rho.clamp_min(self.rho_floor) * T.clamp_min(self.temp_floor)) + 0.5
        tau_therm = 0.5 + (tau_mom - 0.5) / max(self.Pr, 1e-6)
        beta1 = 1.0 / (2.0 * tau_mom)
        beta2 = 1.0 / (2.0 * tau_therm)
        return beta1.clamp(min=1e-4, max=0.999), beta2.clamp(min=1e-4, max=0.999)

    def collision(self, F, G, Feq, Geq, rho, T):
        beta1, beta2 = self.get_relaxation_parameters(rho, T)
        alpha = estimate_entropic_alpha(F, Feq, self.weights, beta1)
        F_post = F + alpha.unsqueeze(0) * beta1.unsqueeze(0) * (Feq - F)
        G_post = G + alpha.unsqueeze(0) * beta2.unsqueeze(0) * (Geq - G)
        F_post = torch.nan_to_num(F_post, nan=self.pop_floor, posinf=1.0, neginf=self.pop_floor).clamp_min(
            self.pop_floor
        )
        G_post = torch.nan_to_num(G_post, nan=self.pop_floor, posinf=1.0, neginf=self.pop_floor).clamp_min(
            self.pop_floor
        )
        return F_post, G_post

    def interpolate_domain(self, populations):
        shift_x = float(self.Uax)
        shift_y = float(self.Uay)
        interpolated = populations

        if abs(shift_x) > 1e-12:
            out = torch.zeros_like(interpolated)
            if shift_x > 0.0:
                div = 1.0 + 2.0 * shift_x
                out[..., 1:] = interpolated[..., 1:] * (1.0 - shift_x) + interpolated[..., :-1] * shift_x
                out[..., 0] = (interpolated[..., 1] * shift_x + interpolated[..., 0] * (1.0 + shift_x)) / div
            else:
                frac = abs(shift_x)
                div = 1.0 + 2.0 * frac
                out[..., :-1] = interpolated[..., :-1] * (1.0 - frac) + interpolated[..., 1:] * frac
                out[..., -1] = (interpolated[..., -2] * frac + interpolated[..., -1] * (1.0 + frac)) / div
            interpolated = out

        if abs(shift_y) > 1e-12:
            out = torch.zeros_like(interpolated)
            if shift_y > 0.0:
                div = 1.0 + 2.0 * shift_y
                out[:, 1:, :] = interpolated[:, 1:, :] * (1.0 - shift_y) + interpolated[:, :-1, :] * shift_y
                out[:, 0, :] = (interpolated[:, 1, :] * shift_y + interpolated[:, 0, :] * (1.0 + shift_y)) / div
            else:
                frac = abs(shift_y)
                div = 1.0 + 2.0 * frac
                out[:, :-1, :] = interpolated[:, :-1, :] * (1.0 - frac) + interpolated[:, 1:, :] * frac
                out[:, -1, :] = (interpolated[:, -2, :] * frac + interpolated[:, -1, :] * (1.0 + frac)) / div
            interpolated = out

        return interpolated

    def shift_operator(self, populations):
        return populations[self.q_indices, self.Y_indices, self.X_indices]

    def streaming(self, F_pos_coll, G_pos_coll):
        Fo = self.interpolate_domain(F_pos_coll)
        Go = self.interpolate_domain(G_pos_coll)
        return self.shift_operator(Fo), self.shift_operator(Go)

    def _apply_wall_reconstruction(self, Fi, Gi, rho, ux, uy, T, Feq, Geq, lambdas):
        if self.wall_mode != "no_slip":
            raise NotImplementedError("Only no-slip Schardin walls are implemented in this paper-aligned path.")

        adjacent_mask = self.wall_missing_mask.any(dim=0)
        rows, cols = torch.where(adjacent_mask)
        if rows.numel() == 0:
            return Fi, Gi

        rho_sel = rho[rows, cols]
        ux_sel = torch.zeros_like(rho_sel)
        uy_sel = torch.zeros_like(rho_sel)
        if self.wall_thermal == "adiabatic":
            T_sel = T[rows, cols]
        elif self.wall_thermal == "isothermal":
            T_sel = torch.full_like(rho_sel, self.wall_T)
        else:
            raise ValueError(f"Unsupported wall thermal BC '{self.wall_thermal}'.")

        lambda_sel = lambdas[:, rows, cols]
        Feq_wall, _ = self.compute_equilibrium(rho_sel, ux_sel, uy_sel, T_sel, lambdas=lambda_sel)
        Geq_wall = compute_geq_from_feq(Feq_wall, T_sel, self.Cv, dimensions=self.dimensions)

        for q in range(self.Qn):
            qmask = self.wall_missing_mask[q, rows, cols]
            if not torch.any(qmask):
                continue
            qr = rows[qmask]
            qc = cols[qmask]
            reflected_f = Fi[self.opp[q], qr, qc] - Feq[self.opp[q], qr, qc]
            reflected_g = Gi[self.opp[q], qr, qc] - Geq[self.opp[q], qr, qc]
            Fi[q, qr, qc] = torch.clamp_min(Feq_wall[q, qmask] + reflected_f, self.pop_floor)
            Gi[q, qr, qc] = torch.clamp_min(Geq_wall[q, qmask] + reflected_g, self.pop_floor)

        Fi[:, self.Obs] = self.wall_Feq_vector[:, None]
        Gi[:, self.Obs] = self.wall_Geq_vector[:, None]
        return Fi, Gi

    def apply_boundaries(self, Fi, Gi, prev_F, prev_G, rho, ux, uy, T, Feq, Geq, lambdas):
        Fi_bc = Fi.clone()
        Gi_bc = Gi.clone()

        Fi_bc[:, :, 0] = self.inlet_Feq
        Gi_bc[:, :, 0] = self.inlet_Geq
        Fi_bc[:, :, self.X - 1] = prev_F[:, :, self.X - 1]
        Gi_bc[:, :, self.X - 1] = prev_G[:, :, self.X - 1]
        Fi_bc[:, 0, :] = prev_F[:, 0, :]
        Gi_bc[:, 0, :] = prev_G[:, 0, :]
        Fi_bc[:, self.Y - 1, :] = prev_F[:, self.Y - 1, :]
        Gi_bc[:, self.Y - 1, :] = prev_G[:, self.Y - 1, :]

        Fi_bc, Gi_bc = self._apply_wall_reconstruction(Fi_bc, Gi_bc, rho, ux, uy, T, Feq, Geq, lambdas)
        return Fi_bc, Gi_bc


def plot_fields(rho, ux, uy, T, step, output_path, obstacle_mask, title, shock_mach, re_value):
    pressure = rho * T
    mach = np.sqrt((ux**2 + uy**2) / np.maximum(1.4 * T, 1e-8))
    fields = [
        ("Density", rho),
        ("Pressure", pressure),
        ("Velocity x", ux),
        ("Mach", mach),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (name, field) in zip(axes.flat, fields):
        im = ax.imshow(field, cmap="jet", origin="upper")
        ax.contour(obstacle_mask.astype(float), levels=[0.5], colors="black", linewidths=0.8)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{title} step {step}\nShock Ma={shock_mach:.3g}, Re={re_value:.3g}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_case_config(config_path, scale):
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    paper_resolution = config["paper_resolution"]
    geometry = config["geometry"]
    domain = config["domain"]
    shock = config["shock"]

    wedge_side = _scaled_positive_int(paper_resolution["wedge_side_points"], scale, minimum=24)
    X = _scaled_positive_int(domain["x_over_wedge_side"] * paper_resolution["wedge_side_points"], scale, minimum=128)
    Y = _scaled_positive_int(domain["y_over_wedge_side"] * paper_resolution["wedge_side_points"], scale, minimum=128)
    return config, {
        "X": X,
        "Y": Y,
        "wedge_side": wedge_side,
        "apex_x": geometry["apex_x_over_side"] * wedge_side,
        "center_y": geometry["center_y_over_side"] * wedge_side,
        "shock_x": shock["initial_x_over_side"] * wedge_side,
        "shock_half_width": max(float(shock["half_width_cells"]) * float(scale), 1.0),
        "apex_points_right": bool(geometry.get("apex_points_right", True)),
    }


def main():
    parser = argparse.ArgumentParser(description="Schardin Problem D D2Q49 entropic wedge solver")
    parser.add_argument("--config", type=str, default="schardin_param.yml")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform grid scaling factor. 1.0 reproduces the paper side-length resolution (300 cells).",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--plot_every", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--output_tag", type=str, default="", help="Optional suffix for the image output folder")
    parser.add_argument("--save_h5", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    dtype = resolve_torch_dtype(args.dtype)
    device = get_device(args.device)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
    config, scaled = _load_case_config(config_path, args.scale)

    solver = SchardinELBMSolver(
        X=scaled["X"],
        Y=scaled["Y"],
        wedge_side=scaled["wedge_side"],
        apex_x=scaled["apex_x"],
        center_y=scaled["center_y"],
        shock_x=scaled["shock_x"],
        shock_half_width=scaled["shock_half_width"],
        shock_mach=float(config["shock"]["mach"]),
        Re=float(config["flow"]["Re_wedge"]),
        rho_pre=float(config["gas"]["rho_pre"]),
        p_pre=float(config["gas"]["p_pre"]),
        vuy=float(config["gas"]["gamma"]),
        Pr=float(config["gas"]["Pr"]),
        shift_fraction=float(config["lattice"]["shift_fraction"]),
        wall_rho=float(config["wall"]["rho"]),
        wall_T=float(config["wall"]["T"]),
        wall_mode=str(config["boundary_conditions"]["wall"]),
        wall_thermal=str(config["boundary_conditions"].get("wall_thermal", "adiabatic")),
        apex_points_right=scaled["apex_points_right"],
        newton_iters=int(config["lattice"].get("newton_iters", 8)),
        newton_tolerance=float(config["lattice"].get("newton_tolerance", 1e-8)),
        device=device,
        dtype=dtype,
    )

    print(
        "Running Schardin Problem D on "
        f"{device}: X={solver.X}, Y={solver.Y}, side={solver.wedge_side}, Q={solver.Qn}, "
        f"shift=({solver.Uax:.4f},{solver.Uay:.4f}), shock_x={solver.shock_x:.1f}, "
        f"Ma_s={solver.shock_mach}, Re={solver.Re}, rho_post={solver.post_state['rho']:.5f}, "
        f"ux_post={solver.post_state['ux']:.5f}"
    )

    Fi0, Gi0, lambdas = solver.initial_conditions()
    image_dir_name = f"schardin_problem_D_side{solver.wedge_side}_steps{args.steps}"
    if args.output_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.output_tag.strip())
        image_dir_name = f"{image_dir_name}_{safe_tag}"
    image_dir = os.path.join(base_dir, "images", image_dir_name)
    os.makedirs(image_dir, exist_ok=True)

    histories = []
    with torch.no_grad():
        for step in tqdm(range(args.steps), desc="Schardin rollout"):
            rho, ux, uy, T = solver.get_macroscopic(Fi0, Gi0)
            rho, ux, uy, T = solver.sanitize_macroscopic(rho, ux, uy, T)
            Feq, lambdas = solver.compute_equilibrium(rho, ux, uy, T, lambdas=lambdas)
            Geq = compute_geq_from_feq(Feq, T, solver.Cv, dimensions=solver.dimensions)
            F_post, G_post = solver.collision(Fi0, Gi0, Feq, Geq, rho, T)
            Fi, Gi = solver.streaming(F_post, G_post)
            Fi0, Gi0 = solver.apply_boundaries(Fi, Gi, Fi0, Gi0, rho, ux, uy, T, Feq, Geq, lambdas)

            rho_np = detach(rho)
            ux_np = detach(ux)
            uy_np = detach(uy)
            T_np = detach(T)
            pressure_np = rho_np * T_np
            histories.append(
                {
                    "step": step,
                    "rho_min": float(np.min(rho_np)),
                    "rho_max": float(np.max(rho_np)),
                    "p_min": float(np.min(pressure_np)),
                    "p_max": float(np.max(pressure_np)),
                    "T_min": float(np.min(T_np)),
                    "T_max": float(np.max(T_np)),
                    "ux_min": float(np.min(ux_np)),
                    "ux_max": float(np.max(ux_np)),
                }
            )

            if step == 0 or step == args.steps - 1 or ((step + 1) % args.plot_every == 0):
                output_path = os.path.join(image_dir, f"fields_step_{step:04d}.png")
                plot_fields(
                    rho_np,
                    ux_np,
                    uy_np,
                    T_np,
                    step,
                    output_path,
                    detach(solver.Obs),
                    "Schardin Problem D",
                    solver.shock_mach,
                    solver.Re,
                )

            if not np.isfinite(rho_np).all() or not np.isfinite(T_np).all():
                print(f"Non-finite field detected at step {step}. Stopping rollout.")
                break

    if args.save_h5:
        with h5py.File(os.path.join(image_dir, "rollout.h5"), "w") as h5f:
            h5f.create_dataset("Fi0", data=detach(Fi0))
            h5f.create_dataset("Gi0", data=detach(Gi0))
            h5f.create_dataset("obstacle", data=detach(solver.Obs))
            h5f.create_dataset(
                "histories",
                data=np.array(
                    [
                        (
                            item["step"],
                            item["rho_min"],
                            item["rho_max"],
                            item["p_min"],
                            item["p_max"],
                            item["T_min"],
                            item["T_max"],
                            item["ux_min"],
                            item["ux_max"],
                        )
                        for item in histories
                    ],
                    dtype=np.float64,
                ),
            )

    final = histories[-1]
    print(
        f"Final step {final['step']}: "
        f"rho in [{final['rho_min']:.5f}, {final['rho_max']:.5f}], "
        f"p in [{final['p_min']:.5f}, {final['p_max']:.5f}], "
        f"T in [{final['T_min']:.5f}, {final['T_max']:.5f}], "
        f"ux in [{final['ux_min']:.5f}, {final['ux_max']:.5f}]"
    )
    print(f"Outputs saved under {image_dir}")


if __name__ == "__main__":
    main()
