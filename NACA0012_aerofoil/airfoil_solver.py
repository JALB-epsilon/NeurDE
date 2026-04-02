import argparse
import os
import re
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src import *
from utilities import detach, get_device, resolve_torch_dtype

matplotlib.use("Agg")


def _as_solver_tensor(value, dtype, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _parse_shift_fraction(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("u_inf", "").replace("u∞", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        raise ValueError(f"Could not parse shifted velocity fraction from {value!r}")
    return float(match.group(0))


def _naca0012_polygon(chord, head_x, center_y, num_points=1200):
    x = np.linspace(0.0, 1.0, num_points)
    yt = 5.0 * 0.12 * (
        0.2969 * np.sqrt(np.clip(x, 1e-12, None))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    upper = np.column_stack([head_x + chord * x, center_y + chord * yt])
    lower = np.column_stack([head_x + chord * x[::-1], center_y - chord * yt[::-1]])
    polygon = np.vstack([upper, lower])
    return polygon


class NACA0012AirfoilBase(nn.Module):
    def __init__(
        self,
        X,
        Y,
        chord,
        head_x_over_C,
        Qn=9,
        Ma0=1.5,
        Re=1.0e4,
        rho0=1.0,
        T0=0.2,
        alpha1=1.35,
        alpha01=1.05,
        eps_low=1.0e-2,
        eps_mid=5.0e-2,
        eps_clamp=1.0,
        tau_cap=1.0,
        vuy=1.4,
        Pr=0.71,
        shift_fraction=0.4,
        device="cuda:0",
        dtype=torch.float32,
    ):
        super().__init__()
        self.X = int(X)
        self.Y = int(Y)
        self.chord = int(chord)
        self.head_x_over_C = float(head_x_over_C)
        self.head_x = float(self.head_x_over_C * self.chord)
        self.center_y = 0.5 * (self.Y - 1)
        self.Qn = int(Qn)
        self.Ma0 = float(Ma0)
        self.Re = float(Re)
        self.rho0 = float(rho0)
        self.T0 = float(T0)
        self.alpha1 = float(alpha1)
        self.alpha01 = float(alpha01)
        self.eps_low = float(eps_low)
        self.eps_mid = float(eps_mid)
        self.eps_clamp = float(eps_clamp)
        self.tau_cap = float(tau_cap)
        self.vuy = float(vuy)
        self.Pr = float(Pr)
        self.shift_fraction = float(shift_fraction)
        self.device = device
        self.dtype = resolve_torch_dtype(dtype)

        ex_values = [1, 0, -1, 0, 1, -1, -1, 1, 0]
        ey_values = [0, 1, 0, -1, 1, 1, -1, -1, 0]
        opp_values = [2, 3, 0, 1, 6, 7, 4, 5, 8]
        self.get_shift_constants()
        self.ex = torch.tensor(ex_values, dtype=self.dtype, device=self.device) + self.Uax
        self.ey = torch.tensor(ey_values, dtype=self.dtype, device=self.device) + self.Uay
        self.ex1 = torch.tensor(ex_values, dtype=self.dtype, device=self.device)
        self.ey1 = torch.tensor(ey_values, dtype=self.dtype, device=self.device)
        self.opp = torch.tensor(opp_values, dtype=torch.long, device=self.device)
        self.get_derived_quantities()
        self.create_obstacle()

    def get_shift_constants(self):
        self.cs0 = np.sqrt(self.vuy * self.T0)
        self.U0 = self.Ma0 * self.cs0
        self.Uax = self.U0 * self.shift_fraction
        self.Uay = 0.0

    def get_derived_quantities(self):
        self.iCv = self.vuy - 1.0
        self.Cp = self.vuy / self.iCv
        self.Cv = 1.0 / self.iCv
        self.R = self.Cp - self.Cv

        self.ex2 = self.ex**2
        self.ey2 = self.ey**2
        self.exey = self.ex * self.ey

        self.shifts_y = -self.ey1.int()
        self.shifts_x = self.ex1.int()
        self.q_indices = torch.arange(self.Qn, device=self.device)[:, None, None]
        y_indices = (torch.arange(self.Y, device=self.device)[None, :, None] - self.shifts_y[:, None, None]) % self.Y
        x_indices = (torch.arange(self.X, device=self.device)[None, None, :] - self.shifts_x[:, None, None]) % self.X
        self.Y_indices = y_indices.expand(self.Qn, self.Y, self.X)
        self.X_indices = x_indices.expand(self.Qn, self.Y, self.X)

        self.muy = self.U0 * self.chord / self.Re

    def create_obstacle(self):
        x_coords = np.arange(self.X, dtype=np.float64) + 0.5
        y_coords = np.arange(self.Y - 1, -1, -1, dtype=np.float64) + 0.5
        xx, yy = np.meshgrid(x_coords, y_coords)
        polygon = _naca0012_polygon(self.chord, self.head_x, self.center_y)
        mask = Path(polygon).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(self.Y, self.X)
        obs = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        obs[:, torch.sum(obs, dim=0) < 2] = 0
        obs[torch.sum(obs, dim=1) < 2, :] = 0
        self.Obs = obs
        self.Fluid = ~obs
        # Fluid nodes whose upwind source along direction q lies inside the solid.
        # These are the links that need simple bounce-back after streaming.
        self.fluid_bounce_mask = self.Obs[self.Y_indices, self.X_indices] & self.Fluid.unsqueeze(0)

        self.colx = torch.arange(self.X, device=self.device)
        self.coly = torch.arange(self.Y, device=self.device)
        self.inlet_rows = self.coly
        self.inlet_cols = torch.zeros_like(self.inlet_rows)
        self.outlet_rows = self.coly
        self.outlet_cols = torch.full((self.Y,), self.X - 1, dtype=torch.long, device=self.device)
        self.top_rows = torch.zeros(self.X, dtype=torch.long, device=self.device)
        self.top_cols = self.colx
        self.bottom_rows = torch.full((self.X,), self.Y - 1, dtype=torch.long, device=self.device)
        self.bottom_cols = self.colx

    def dot_prod(self, ux, uy):
        return ux**2 + uy**2

    def get_energy_from_temp(self, ux, uy, T):
        return T * self.Cv + 0.5 * self.dot_prod(ux, uy)

    def get_temp_from_energy(self, ux, uy, E):
        return self.iCv * (E - 0.5 * self.dot_prod(ux, uy))

    def get_density(self, F):
        pop_dim = 1 if F.dim() == 4 else 0
        return torch.sum(F, dim=pop_dim).to(self.device)

    def get_momentum(self, F):
        if F.dim() == 4:
            rho_ux = torch.sum(F * self.ex.view(1, self.Qn, 1, 1), dim=1).to(self.device)
            rho_uy = torch.sum(F * self.ey.view(1, self.Qn, 1, 1), dim=1).to(self.device)
            return rho_ux, rho_uy
        rho_ux = torch.tensordot(self.ex, F, dims=([0], [0])).to(self.device)
        rho_uy = torch.tensordot(self.ey, F, dims=([0], [0])).to(self.device)
        return rho_ux, rho_uy

    def get_energy_density(self, G):
        pop_dim = 1 if G.dim() == 4 else 0
        return torch.sum(G, dim=pop_dim).to(self.device)

    def get_macroscopic(self, F, G):
        rho = self.get_density(F)
        inv_rho = 1.0 / rho.clamp_min(1e-8)
        rho_ux, rho_uy = self.get_momentum(F)
        ux = rho_ux * inv_rho
        uy = rho_uy * inv_rho
        E = 0.5 * self.get_energy_density(G) * inv_rho
        return rho, ux, uy, E

    def get_w(self, T):
        if T.dim() == 3:
            w = torch.zeros((T.shape[0], self.Qn, self.Y, self.X), device=self.device, dtype=T.dtype)
        else:
            w = torch.zeros((self.Qn, self.Y, self.X), device=self.device, dtype=T.dtype)
        one_minus_T = 1.0 - T
        w[..., :4, :, :] = (one_minus_T * T * 0.5).unsqueeze(-3)
        w[..., 4:8, :, :] = (T**2 * 0.25).unsqueeze(-3)
        w[..., 8, :, :] = one_minus_T**2
        return w

    def get_local_Mach(self, ux, uy, T):
        return torch.sqrt(self.dot_prod(ux, uy) / (self.vuy * T.clamp_min(1e-8)))

    def get_nonequilibrium_sensor(self, F, Feq):
        pop_dim = 1 if F.dim() == 4 else 0
        neq_l1 = torch.sum(torch.abs(F - Feq), dim=pop_dim)
        eq_l1 = torch.sum(torch.abs(Feq), dim=pop_dim).clamp_min(1e-8)
        return neq_l1 / eq_l1

    def get_stabilization_alpha(self, eps, tau_dl):
        alpha = torch.ones_like(eps)
        alpha = torch.where(eps >= self.eps_low, eps.new_tensor(self.alpha01), alpha)
        alpha = torch.where(eps >= self.eps_mid, eps.new_tensor(self.alpha1), alpha)
        # In the strongest nonequilibrium cells, replace populations by
        # equilibrium in one BGK step: tau = 1.
        alpha_eq = self.tau_cap / tau_dl.clamp_min(1e-8)
        alpha = torch.where(eps >= self.eps_clamp, alpha_eq, alpha)
        return alpha

    def _stabilized_tau(self, tau_dl, eps):
        # Follow the paper's proof-of-concept stabilization more literally:
        # tau(eps) = tau * alpha(eps), with alpha piecewise constant in eps.
        alpha = self.get_stabilization_alpha(eps, tau_dl)
        tau = tau_dl * alpha
        return tau.clamp_min(0.500001).clamp_max(self.tau_cap), alpha

    def get_relaxation_diagnostics(self, rho, ux, uy, T, F, Feq):
        tau_dl = self.muy / (rho.clamp_min(1e-8) * T.clamp_min(1e-8)) + 0.5
        eps = self.get_nonequilibrium_sensor(F, Feq)
        tau, alpha = self._stabilized_tau(tau_dl, eps)
        tau_t = 0.5 + (tau - 0.5) / self.Pr
        return {
            "eps": eps,
            "alpha": alpha,
            "tau_dl": tau_dl,
            "tau": tau,
            "tau_t": tau_t,
        }

    def get_relaxation_time(self, rho, ux, uy, T, F, Feq):
        diagnostics = self.get_relaxation_diagnostics(rho, ux, uy, T, F, Feq)
        tau = diagnostics["tau"]
        tau_t = diagnostics["tau_t"]
        if F.dim() == 4:
            tau = tau.unsqueeze(1).expand(-1, self.Qn, self.Y, self.X)
        else:
            tau = tau.reshape(1, self.Y, self.X).expand(self.Qn, self.Y, self.X)
            tau_t = tau_t.reshape(1, self.Y, self.X).expand(self.Qn, self.Y, self.X)
        if F.dim() == 4:
            tau_t = tau_t.unsqueeze(1).expand(-1, self.Qn, self.Y, self.X)
        return 1.0 / tau, 1.0 / tau_t

    def get_Feq(self, rho, ux, uy, T):
        return F_pop_torch.compute_Feq(rho, ux, self.Uax, uy, self.Uay, T)

    def get_Feq_obs(self, rho, ux, uy, T):
        return F_pop_torch.compute_Feq_obstacle(rho, ux, self.Uax, uy, self.Uay, T, obstacle=self.Obs)

    def get_Feq_BC(self, rho, ux, uy, T, row, col):
        return F_pop_torch.compute_Feq_BC(rho, ux, self.Uax, uy, self.Uay, T, row, col)

    def get_Geq_Newton_solver(self, rho, ux, uy, T, khi, zetax, zetay):
        dtype = rho.dtype
        return levermore_Geq_torch(
            self.ex,
            self.ey,
            ux,
            uy,
            T,
            rho,
            self.Cv,
            self.Qn,
            _as_solver_tensor(khi, dtype, self.device),
            _as_solver_tensor(zetax, dtype, self.device),
            _as_solver_tensor(zetay, dtype, self.device),
            device=self.device,
        )

    def get_Geq_Newton_solver_obs(self, rho, ux, uy, T, khi, zetax, zetay):
        dtype = rho.dtype
        return levermore_Geq_Obs_torch(
            self.ex,
            self.ey,
            ux,
            uy,
            T,
            rho,
            self.Cv,
            self.Qn,
            _as_solver_tensor(khi, dtype, self.device),
            _as_solver_tensor(zetax, dtype, self.device),
            _as_solver_tensor(zetay, dtype, self.device),
            self.Obs,
            device=self.device,
        )

    def get_Geq_Newton_solver_BC(self, rho, ux, uy, T, khi, zetax, zetay, row, col):
        dtype = rho.dtype
        return levermore_Geq_BCs_torch(
            self.ex,
            self.ey,
            ux,
            uy,
            T,
            rho,
            self.Cv,
            self.Qn,
            _as_solver_tensor(khi, dtype, self.device),
            _as_solver_tensor(zetax, dtype, self.device),
            _as_solver_tensor(zetay, dtype, self.device),
            row,
            col,
            device=self.device,
        )

    def get_qs(self, F, rho, ux, uy, T):
        p_eq_xx = rho * ux * ux + rho * T
        p_eq_yy = rho * uy * uy + rho * T
        p_eq_xy = rho * ux * uy
        p_xx = torch.tensordot(self.ex2, F, dims=([0], [0])).to(self.device)
        p_yy = torch.tensordot(self.ey2, F, dims=([0], [0])).to(self.device)
        p_xy = torch.tensordot(self.exey, F, dims=([0], [0])).to(self.device)
        diff_xy = p_xy - p_eq_xy
        qsx = 2.0 * ux * (p_xx - p_eq_xx) + 2.0 * uy * diff_xy
        qsy = 2.0 * uy * (p_yy - p_eq_yy) + 2.0 * ux * diff_xy
        return qsx, qsy

    def from_macro_to_lattice_Gis(self, F, rho, ux, uy, T):
        w = self.get_w(T)
        qsx, qsy = self.get_qs(F, rho, ux, uy, T)
        return w * (qsx * self.ex[:, None, None] + qsy * self.ey[:, None, None]) / T.clamp_min(1e-8)[None, :, :]

    def interpolate_domain(self, Fo, Go):
        div = 1.0 + 2.0 * self.Uax
        Fo1 = torch.zeros_like(Fo)
        Go1 = torch.zeros_like(Go)
        Fo1[..., 1 : self.X] = Fo[..., 1 : self.X] * (1.0 - self.Uax) + Fo[..., 0 : self.X - 1] * self.Uax
        Go1[..., 1 : self.X] = Go[..., 1 : self.X] * (1.0 - self.Uax) + Go[..., 0 : self.X - 1] * self.Uax
        Fo1[..., 0] = (Fo[..., 1] * self.Uax + Fo[..., 0] * (1.0 + self.Uax)) / div
        Go1[..., 0] = (Go[..., 1] * self.Uax + Go[..., 0] * (1.0 + self.Uax)) / div
        return Fo1, Go1

    def collision(self, F, G, Feq, Geq, rho, ux, uy, T):
        omega, omegaT = self.get_relaxation_time(rho, ux, uy, T, F, Feq)
        Gis = self.from_macro_to_lattice_Gis(F, rho, ux, uy, T)
        F_pos = F - omega * (F - Feq)
        G_pos = G - omega * (G - Geq) + (omega - omegaT) * Gis
        return F_pos, G_pos

    def shift_operator(self, F, G):
        Fi = F[self.q_indices, self.Y_indices, self.X_indices]
        Gi = G[self.q_indices, self.Y_indices, self.X_indices]
        return Fi, Gi

    def streaming(self, F_pos_coll, G_pos_coll):
        Fo1, Go1 = self.interpolate_domain(F_pos_coll, G_pos_coll)
        return self.shift_operator(Fo1, Go1)

    def initial_conditions(self):
        rho = torch.full((self.Y, self.X), self.rho0, device=self.device, dtype=self.dtype)
        ux = torch.full((self.Y, self.X), self.U0, device=self.device, dtype=self.dtype)
        uy = torch.zeros((self.Y, self.X), device=self.device, dtype=self.dtype)
        T = torch.full((self.Y, self.X), self.T0, device=self.device, dtype=self.dtype)
        khi0 = torch.zeros((self.Y, self.X), device=self.device, dtype=self.dtype)
        zetax0 = torch.zeros((self.Y, self.X), device=self.device, dtype=self.dtype)
        zetay0 = torch.zeros((self.Y, self.X), device=self.device, dtype=self.dtype)
        Fi0 = self.get_Feq(rho, ux, uy, T)
        Gi0, khi, zetax, zetay = self.get_Geq_Newton_solver(rho, ux, uy, T, khi0, zetax0, zetay0)
        return Fi0.to(self.device), Gi0.to(self.device), khi, zetax, zetay

    def get_body_distribution(self, rho, ux, uy, T, khi, zetax, zetay):
        zero = torch.zeros((), device=self.device, dtype=ux.dtype)
        ux_body = torch.where(self.Obs, zero, ux)
        uy_body = torch.where(self.Obs, zero, uy)
        rho_fill = torch.full((), self.rho0, device=self.device, dtype=rho.dtype)
        temp_fill = torch.full((), self.T0, device=self.device, dtype=T.dtype)
        rho_body = torch.where(self.Obs, rho_fill, rho)
        T_body = torch.where(self.Obs, temp_fill, T.clamp_min(1e-6))
        Fi_body = self.get_Feq_obs(rho_body, ux_body, uy_body, T_body)
        Gi_body, _, _, _ = self.get_Geq_Newton_solver_obs(rho_body, ux_body, uy_body, T_body, khi, zetax, zetay)
        return Fi_body, Gi_body

    def _prepare_boundary_macro_state(self, rho, ux, uy, T):
        rho_bc = rho.clone()
        ux_bc = ux.clone()
        uy_bc = uy.clone()
        T_bc = T.clone()

        rho_bc[self.inlet_rows, 0] = self.rho0
        ux_bc[self.inlet_rows, 0] = self.U0
        uy_bc[self.inlet_rows, 0] = 0.0
        T_bc[self.inlet_rows, 0] = self.T0

        rho_bc[0, self.colx] = self.rho0
        ux_bc[0, self.colx] = self.U0
        uy_bc[0, self.colx] = 0.0
        T_bc[0, self.colx] = self.T0

        rho_bc[self.Y - 1, self.colx] = self.rho0
        ux_bc[self.Y - 1, self.colx] = self.U0
        uy_bc[self.Y - 1, self.colx] = 0.0
        T_bc[self.Y - 1, self.colx] = self.T0

        # First-order Neumann outlet on macros, matching the paper's BC family.
        rho_bc[self.outlet_rows, self.X - 1] = rho_bc[self.outlet_rows, self.X - 2]
        ux_bc[self.outlet_rows, self.X - 1] = ux_bc[self.outlet_rows, self.X - 2]
        uy_bc[self.outlet_rows, self.X - 1] = uy_bc[self.outlet_rows, self.X - 2]
        T_bc[self.outlet_rows, self.X - 1] = T_bc[self.outlet_rows, self.X - 2]
        return rho_bc, ux_bc, uy_bc, T_bc

    def get_boundary_distributions(self, rho, ux, uy, T, khi, zetax, zetay):
        rho_bc, ux_bc, uy_bc, T_bc = self._prepare_boundary_macro_state(rho, ux, uy, T)
        inlet_F = self.get_Feq_BC(rho_bc, ux_bc, uy_bc, T_bc, self.inlet_rows, self.inlet_cols)
        inlet_G, _, _, _ = self.get_Geq_Newton_solver_BC(
            rho_bc, ux_bc, uy_bc, T_bc, khi, zetax, zetay, self.inlet_rows, self.inlet_cols
        )

        top_F = self.get_Feq_BC(rho_bc, ux_bc, uy_bc, T_bc, self.top_rows, self.top_cols)
        top_G, _, _, _ = self.get_Geq_Newton_solver_BC(
            rho_bc, ux_bc, uy_bc, T_bc, khi, zetax, zetay, self.top_rows, self.top_cols
        )

        bottom_F = self.get_Feq_BC(rho_bc, ux_bc, uy_bc, T_bc, self.bottom_rows, self.bottom_cols)
        bottom_G, _, _, _ = self.get_Geq_Newton_solver_BC(
            rho_bc, ux_bc, uy_bc, T_bc, khi, zetax, zetay, self.bottom_rows, self.bottom_cols
        )

        outlet_F = self.get_Feq_BC(rho_bc, ux_bc, uy_bc, T_bc, self.outlet_rows, self.outlet_cols)
        outlet_G, _, _, _ = self.get_Geq_Newton_solver_BC(
            rho_bc, ux_bc, uy_bc, T_bc, khi, zetax, zetay, self.outlet_rows, self.outlet_cols
        )
        return inlet_F, inlet_G, top_F, top_G, bottom_F, bottom_G, outlet_F, outlet_G

    def enforce_body_and_bc(
        self,
        Fi,
        Gi,
        F_post,
        G_post,
        Fi_body,
        Gi_body,
        inlet_F,
        inlet_G,
        top_F,
        top_G,
        bottom_F,
        bottom_G,
        outlet_F,
        outlet_G,
    ):
        Fi_obs = Fi.clone()
        Gi_obs = Gi.clone()

        # Simple link-wise bounce-back: for fluid nodes adjacent to the airfoil,
        # replace incoming populations from solid links with the reflected
        # post-collision population from the opposite direction.
        for q in range(self.Qn):
            mask = self.fluid_bounce_mask[q]
            if torch.any(mask):
                Fi_obs[q, mask] = F_post[self.opp[q], mask]
                Gi_obs[q, mask] = G_post[self.opp[q], mask]

        Fi_obs[:, self.Obs] = Fi_body
        Gi_obs[:, self.Obs] = Gi_body
        Fi_obs[:, self.inlet_rows, 0] = inlet_F
        Gi_obs[:, self.inlet_rows, 0] = inlet_G
        Fi_obs[:, self.top_rows, self.top_cols] = top_F
        Gi_obs[:, self.top_rows, self.top_cols] = top_G
        Fi_obs[:, self.bottom_rows, self.bottom_cols] = bottom_F
        Gi_obs[:, self.bottom_rows, self.bottom_cols] = bottom_G
        Fi_obs[:, self.outlet_rows, self.outlet_cols] = outlet_F
        Gi_obs[:, self.outlet_rows, self.outlet_cols] = outlet_G
        return Fi_obs, Gi_obs


def _plot_window(obstacle_mask):
    ys, xs = np.where(obstacle_mask)
    if xs.size == 0:
        y_size, x_size = obstacle_mask.shape
        return 0, x_size, 0, y_size, 1.0, 0.0, 0.5 * (y_size - 1)
    x_lead = float(xs.min())
    x_trail = float(xs.max())
    y_center = 0.5 * (float(ys.min()) + float(ys.max()))
    chord_px = max(1.0, x_trail - x_lead + 1.0)
    x_start = max(0, int(np.floor(x_lead - 0.35 * chord_px)))
    x_stop = min(obstacle_mask.shape[1], int(np.ceil(x_lead + 6.0 * chord_px)))
    y_start = max(0, int(np.floor(y_center - 1.25 * chord_px)))
    y_stop = min(obstacle_mask.shape[0], int(np.ceil(y_center + 1.25 * chord_px)))
    return x_start, x_stop, y_start, y_stop, chord_px, x_lead, y_center


def _robust_limits(field, mask, floor=None, symmetric=False):
    values = field[~mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if symmetric:
        bound = max(abs(lo), abs(hi), 1e-8)
        pad = 0.05 * bound
        return -(bound + pad), bound + pad
    if floor is not None:
        lo = max(lo, floor)
    if hi <= lo:
        hi = lo + 1e-6
    pad = 0.03 * (hi - lo)
    return lo - pad, hi + pad


def plot_fields(rho, ux, uy, T, step, output_path, obstacle_mask, title_prefix, Ma0, Re, rho_ref, ux_ref, T_ref):
    mach = np.sqrt((ux**2 + uy**2) / np.maximum(1.4 * T, 1e-8))
    x_start, x_stop, y_start, y_stop, chord_px, x_lead, y_center = _plot_window(obstacle_mask)
    obs_roi = obstacle_mask[y_start:y_stop, x_start:x_stop]
    y_min = (y_center - y_stop) / chord_px
    y_max = (y_center - y_start) / chord_px
    extent = [(x_start - x_lead) / chord_px, (x_stop - x_lead) / chord_px, y_min, y_max]

    fields = [
        ("Density change (%)", 100.0 * (rho / max(rho_ref, 1e-8) - 1.0), "RdBu_r", None, True),
        ("u_x change (%)", 100.0 * (ux / max(abs(ux_ref), 1e-8) - 1.0), "RdBu_r", None, True),
        ("Temperature change (%)", 100.0 * (T / max(T_ref, 1e-8) - 1.0), "RdBu_r", None, True),
        ("Mach change", mach - Ma0, "RdBu_r", None, True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    for ax, (name, field, cmap_name, floor, symmetric) in zip(axes.flat, fields):
        roi = field[y_start:y_stop, x_start:x_stop]
        plot_data = np.ma.array(roi[::-1, :], mask=obs_roi[::-1, :])
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#f4f4f4")
        vmin, vmax = _robust_limits(roi, obs_roi, floor=floor, symmetric=symmetric)
        im = ax.imshow(plot_data, cmap=cmap, origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="auto")
        ax.contour(
            obs_roi[::-1, :].astype(float),
            levels=[0.5],
            colors="black",
            linewidths=0.9,
            origin="lower",
            extent=extent,
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("x / C")
        ax.set_ylabel("y / C")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"{title_prefix} step {step}\nMa={Ma0:.3g}, Re={Re:.3g}",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paper_fields(rho, ux, uy, T, step, output_path, obstacle_mask, title_prefix, Ma0, Re):
    mach = np.sqrt((ux**2 + uy**2) / np.maximum(1.4 * T, 1e-8))
    x_start, x_stop, y_start, y_stop, chord_px, x_lead, y_center = _plot_window(obstacle_mask)
    obs_roi = obstacle_mask[y_start:y_stop, x_start:x_stop]
    y_min = (y_center - y_stop) / chord_px
    y_max = (y_center - y_start) / chord_px
    extent = [(x_start - x_lead) / chord_px, (x_stop - x_lead) / chord_px, y_min, y_max]

    fields = [
        ("Density", rho, "viridis"),
        ("Local Mach", mach, "magma"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for ax, (name, field, cmap_name) in zip(axes.flat, fields):
        roi = field[y_start:y_stop, x_start:x_stop]
        plot_data = np.ma.array(roi[::-1, :], mask=obs_roi[::-1, :])
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#f4f4f4")
        vmin, vmax = _robust_limits(roi, obs_roi, symmetric=False)
        im = ax.imshow(plot_data, cmap=cmap, origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="auto")
        ax.contour(
            obs_roi[::-1, :].astype(float),
            levels=[0.5],
            colors="white" if cmap_name == "magma" else "black",
            linewidths=0.9,
            origin="lower",
            extent=extent,
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("x / C")
        ax.set_ylabel("y / C")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"{title_prefix} paper-style fields step {step}\nMa={Ma0:.3g}, Re={Re:.3g}",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stabilization_fields(eps, alpha, tau, tau_dl, step, output_path, obstacle_mask, title_prefix, Ma0, Re):
    x_start, x_stop, y_start, y_stop, chord_px, x_lead, y_center = _plot_window(obstacle_mask)
    obs_roi = obstacle_mask[y_start:y_stop, x_start:x_stop]
    y_min = (y_center - y_stop) / chord_px
    y_max = (y_center - y_start) / chord_px
    extent = [(x_start - x_lead) / chord_px, (x_stop - x_lead) / chord_px, y_min, y_max]

    log_eps = np.log10(np.maximum(eps, 1e-12))
    tau_added = tau - tau_dl
    fields = [
        ("log10 eps", log_eps, "magma", None, False),
        ("alpha(eps)", alpha, "viridis", (1.0, max(1.0, float(np.nanmax(alpha)))), False),
        ("tau", tau, "viridis", (0.5, max(1.0, float(np.nanmax(tau)))), False),
        ("tau - tau_dl", tau_added, "magma", None, False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), constrained_layout=True)
    for ax, (name, field, cmap_name, fixed_limits, symmetric) in zip(axes.flat, fields):
        roi = field[y_start:y_stop, x_start:x_stop]
        plot_data = np.ma.array(roi[::-1, :], mask=obs_roi[::-1, :])
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#f4f4f4")
        if fixed_limits is None:
            vmin, vmax = _robust_limits(roi, obs_roi, symmetric=symmetric)
        else:
            vmin, vmax = fixed_limits
        im = ax.imshow(plot_data, cmap=cmap, origin="lower", extent=extent, vmin=vmin, vmax=vmax, aspect="auto")
        ax.contour(
            obs_roi[::-1, :].astype(float),
            levels=[0.5],
            colors="white" if cmap_name == "magma" else "black",
            linewidths=0.9,
            origin="lower",
            extent=extent,
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("x / C")
        ax.set_ylabel("y / C")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        f"{title_prefix} stabilization step {step}\nMa={Ma0:.3g}, Re={Re:.3g}",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_scenario(config, scenario_name, chord_override=None, re_override=None):
    scenario = config["benchmark_scenarios"][scenario_name]
    if scenario_name == "transonic":
        chord = int(scenario["domain"]["chord_resolution"] if chord_override is None else chord_override)
        Re = float(scenario["Re"] if re_override is None else re_override)
    else:
        available = list(scenario["domain"]["chord_resolutions"])
        chord = int(available[0] if chord_override is None else chord_override)
        Re = float(scenario["Re_range"]["min"] if re_override is None else re_override)
    domain_x = int(round(chord * float(scenario["domain"]["extent_in_chords"]["x"])))
    domain_y = int(round(chord * float(scenario["domain"]["extent_in_chords"]["y"])))
    return scenario, chord, domain_x, domain_y, Re


def _prepare_output_dir(path, clear_existing=True):
    if clear_existing and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Newton-Geq NACA0012 aerofoil solver pilot")
    parser.add_argument("--config", type=str, default="airfoil_param.yml")
    parser.add_argument("--scenario", type=str, default="supersonic", choices=["transonic", "supersonic"])
    parser.add_argument("--C", type=int, default=None, help="Override chord resolution")
    parser.add_argument("--Re", type=float, default=None, help="Override Reynolds number")
    parser.add_argument("--rho0", type=float, default=1.0, help="Freestream density inference for the pilot run")
    parser.add_argument("--T0", type=float, default=0.2, help="Freestream temperature inference for the pilot run")
    parser.add_argument("--alpha1", type=float, default=1.35, help="Non-equilibrium relaxation multiplier for 0.1 <= eps < 1")
    parser.add_argument("--alpha01", type=float, default=1.05, help="Non-equilibrium relaxation multiplier for 0.01 <= eps < 0.1")
    parser.add_argument("--eps_low", type=float, default=1.0e-2, help="Epsilon threshold where stabilization starts to increase")
    parser.add_argument("--eps_mid", type=float, default=5.0e-2, help="Epsilon threshold for moderate stabilization")
    parser.add_argument("--eps_clamp", type=float, default=1.0, help="Epsilon threshold where tau reaches tau_cap")
    parser.add_argument("--tau_cap", type=float, default=1.0, help="Maximum local relaxation time used in strongly non-equilibrium cells")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--plot_every", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--output_tag", type=str, default="", help="Optional suffix for the image output folder")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow explicit CPU execution when CUDA is unavailable")
    parser.add_argument("--keep_output_dir", action="store_true", help="Keep existing images in the target output folder")
    parser.add_argument("--save_h5", action="store_true")
    args = parser.parse_args()

    dtype = resolve_torch_dtype(args.dtype)
    device = get_device(args.device)
    if not str(device).startswith("cuda") and not args.allow_cpu:
        raise RuntimeError("CUDA is required for this solver run. Re-run with a GPU device or pass --allow_cpu.")
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    scenario, chord, X, Y, Re = _select_scenario(config, args.scenario, args.C, args.Re)
    gamma = float(config["thermodynamics"]["gamma"])
    Pr = float(config["thermodynamics"]["Pr"])
    shift_fraction = _parse_shift_fraction(scenario["shifted_velocity"])
    head_x_over_C = float(scenario["aerofoil_position"]["head_x_over_C"])
    Ma0 = scenario["Ma_inf"][0] if isinstance(scenario["Ma_inf"], list) else float(scenario["Ma_inf"])

    solver = NACA0012AirfoilBase(
        X=X,
        Y=Y,
        chord=chord,
        head_x_over_C=head_x_over_C,
        Qn=int(config["lattice"]["Qn"]),
        Ma0=Ma0,
        Re=Re,
        rho0=args.rho0,
        T0=args.T0,
        alpha1=args.alpha1,
        alpha01=args.alpha01,
        eps_low=args.eps_low,
        eps_mid=args.eps_mid,
        eps_clamp=args.eps_clamp,
        tau_cap=args.tau_cap,
        vuy=gamma,
        Pr=Pr,
        shift_fraction=shift_fraction,
        device=device,
        dtype=dtype,
    )

    print(
        f"Running {args.scenario} NACA0012 Newton-Geq pilot on {device}: "
        f"X={X}, Y={Y}, C={chord}, Ma={Ma0}, Re={Re}, shift={shift_fraction}, "
        f"rho0={args.rho0}, T0={args.T0}, alpha01={args.alpha01}, alpha1={args.alpha1}, "
        f"eps=[{args.eps_low}, {args.eps_mid}, {args.eps_clamp}], tau_cap={args.tau_cap}"
    )

    Fi0, Gi0, khi, zetax, zetay = solver.initial_conditions()
    image_dir_name = f"newton_geq_{args.scenario}_C{chord}_Re{Re:.0e}"
    if args.output_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.output_tag.strip())
        image_dir_name = f"{image_dir_name}_{safe_tag}"
    image_dir = os.path.join("images", image_dir_name)
    _prepare_output_dir(image_dir, clear_existing=not args.keep_output_dir)

    histories = []
    with torch.no_grad():
        for step in tqdm(range(args.steps), desc="Newton-Geq rollout"):
            rho, ux, uy, E = solver.get_macroscopic(Fi0, Gi0)
            T = solver.get_temp_from_energy(ux, uy, E).clamp_min(1e-6)
            Feq = solver.get_Feq(rho, ux, uy, T)
            should_plot = step == 0 or step == args.steps - 1 or ((step + 1) % args.plot_every == 0)
            stabilization = solver.get_relaxation_diagnostics(rho, ux, uy, T, Fi0, Feq) if should_plot else None
            Geq, khi, zetax, zetay = solver.get_Geq_Newton_solver(rho, ux, uy, T, khi, zetax, zetay)
            F_post, G_post = solver.collision(Fi0, Gi0, Feq, Geq, rho, ux, uy, T)
            Fi, Gi = solver.streaming(F_post, G_post)
            Fi_body, Gi_body = solver.get_body_distribution(rho, ux, uy, T, khi, zetax, zetay)
            inlet_F, inlet_G, top_F, top_G, bottom_F, bottom_G, outlet_F, outlet_G = solver.get_boundary_distributions(
                rho, ux, uy, T, khi, zetax, zetay
            )
            Fi0, Gi0 = solver.enforce_body_and_bc(
                Fi,
                Gi,
                F_post,
                G_post,
                Fi_body,
                Gi_body,
                inlet_F,
                inlet_G,
                top_F,
                top_G,
                bottom_F,
                bottom_G,
                outlet_F,
                outlet_G,
            )

            rho_np = detach(rho)
            ux_np = detach(ux)
            uy_np = detach(uy)
            T_np = detach(T)
            histories.append(
                {
                    "step": step,
                    "rho_min": float(np.min(rho_np)),
                    "rho_max": float(np.max(rho_np)),
                    "T_min": float(np.min(T_np)),
                    "T_max": float(np.max(T_np)),
                    "ux_min": float(np.min(ux_np)),
                    "ux_max": float(np.max(ux_np)),
                }
            )

            if should_plot:
                output_path = os.path.join(image_dir, f"fields_step_{step:04d}.png")
                plot_fields(
                    rho_np,
                    ux_np,
                    uy_np,
                    T_np,
                    step,
                    output_path,
                    detach(solver.Obs),
                    f"NACA0012 {args.scenario} Newton-Geq",
                    Ma0,
                    Re,
                    args.rho0,
                    solver.U0,
                    args.T0,
                )
                paper_output_path = os.path.join(image_dir, f"paper_fields_step_{step:04d}.png")
                plot_paper_fields(
                    rho_np,
                    ux_np,
                    uy_np,
                    T_np,
                    step,
                    paper_output_path,
                    detach(solver.Obs),
                    f"NACA0012 {args.scenario} Newton-Geq",
                    Ma0,
                    Re,
                )
                stabilization_path = os.path.join(image_dir, f"stabilization_step_{step:04d}.png")
                plot_stabilization_fields(
                    detach(stabilization["eps"]),
                    detach(stabilization["alpha"]),
                    detach(stabilization["tau"]),
                    detach(stabilization["tau_dl"]),
                    step,
                    stabilization_path,
                    detach(solver.Obs),
                    f"NACA0012 {args.scenario} Newton-Geq",
                    Ma0,
                    Re,
                )

            if not np.isfinite(rho_np).all() or not np.isfinite(T_np).all():
                print(f"Non-finite field detected at step {step}. Stopping rollout.")
                break

    if args.save_h5:
        with h5py.File(os.path.join(image_dir, "pilot_rollout.h5"), "w") as h5f:
            h5f.create_dataset("Fi0", data=detach(Fi0))
            h5f.create_dataset("Gi0", data=detach(Gi0))
            h5f.create_dataset("obstacle", data=detach(solver.Obs))

    final = histories[-1]
    print(
        f"Final finite step {final['step']}: "
        f"rho=[{final['rho_min']:.4e}, {final['rho_max']:.4e}], "
        f"T=[{final['T_min']:.4e}, {final['T_max']:.4e}], "
        f"ux=[{final['ux_min']:.4e}, {final['ux_max']:.4e}]"
    )
    print(f"Plots written to {image_dir}")


if __name__ == "__main__":
    main()
