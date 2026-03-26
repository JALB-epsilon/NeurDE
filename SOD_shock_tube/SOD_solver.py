import torch
import torch.nn as nn
import numpy as np
from src import F_pop_torch, levermore_Geq_torch
from utilities import detach, get_device
from analytic_sod import analytic_reference_from_case

# Note: matmul precision will be set conditionally based on precision argument


class SODSolver(nn.Module):
    def __init__(self, X=3001, Y=5, Qn=9,
                 alpha1=1.2,
                 alpha01=1.05,
                 vuy=2,
                 Pr=0.71,
                 muy=0.025,
                 Uax=0.0,
                 Uay=0.0,
                 device='cuda'):
        super().__init__()
        self.X = X
        self.Y = Y
        self.Qn = Qn
        self.alpha1 = alpha1
        self.alpha01 = alpha01
        self.vuy = vuy
        self.Pr = Pr  # Prandtl number
        self.muy = muy # dynamic viscosity
        self.Uax = Uax 
        self.Uay = Uay
        self.device = device
        self.dtype = torch.get_default_dtype()
        self.is_compiled = False  # Track if methods are compiled
        ex_values = [1, 0, -1, 0, 1, -1, -1, 1, 0]
        ey_values = [0, 1, 0, -1, 1, 1, -1, -1, 0]
        self.ex = torch.tensor(ex_values, dtype=self.dtype, device=self.device) + self.Uax
        self.ey = torch.tensor(ey_values, dtype=self.dtype, device=self.device) + self.Uay
        self.ex1 = torch.tensor(ex_values, dtype=self.dtype, device=self.device)
        self.ey1 = torch.tensor(ey_values, dtype=self.dtype, device=self.device)
        del ex_values, ey_values
        self.Lx = self.X // 2
        self.get_derived_quantities()

    def get_derived_quantities(self): 
        self.iCv = self.vuy - 1 
        self.Cp = self.vuy / self.iCv 
        self.Cv = 1 / self.iCv
        # Pre-calculate for efficiency
        self.ex2 = self.ex**2
        self.ey2 = self.ey**2
        self.exey = self.ex * self.ey
        self.R = self.Cp-self.Cv  #gas constant
        self.shifts_y = -self.ey1.int()
        self.shifts_x = self.ex1.int()
        self.q_indices = torch.arange(self.Qn, device=self.device)[:, None, None]
        Y_indices = (torch.arange(self.Y, device=self.device)[None, :, None] - self.shifts_y[:, None, None]) % self.Y
        X_indices = (torch.arange(self.X, device=self.device)[None, None, :] - self.shifts_x[:, None, None]) % self.X

        self.Y_indices = Y_indices.expand(self.Qn, self.Y, self.X)
        self.X_indices = X_indices.expand(self.Qn, self.Y, self.X)
        del Y_indices, X_indices

    def dot_prod(self, ux, uy):
        return ux**2 + uy**2

    def get_energy_from_temp(self, ux, uy, T):
        uu = self.dot_prod(ux, uy)
        return T * self.Cv + uu / 2
    
    def get_temp_from_energy(self, ux, uy, E):
        uu = self.dot_prod(ux, uy)
        return self.iCv * (E - uu / 2)
    
    def get_heat_flux_Maxwellian(self, rho, ux, uy, E, T):
        H = E + T
        rhoH2 = 2 * rho * H 
        qx = rhoH2 * ux  
        qy = rhoH2 * uy  
        del H, rhoH2
        return qx, qy

    def _velocity_view(self, ndim):
        return (1,) * (ndim - 3) + (self.Qn, 1, 1)
        
    def get_density(self, F): 
        rho = torch.sum(F, dim=-3).to(self.device)
        return rho
    
    def get_momentum(self, F): 
        ex = self.ex.view(self._velocity_view(F.ndim))
        ey = self.ey.view(self._velocity_view(F.ndim))
        rho_ux = torch.sum(ex * F, dim=-3).to(self.device)
        rho_uy = torch.sum(ey * F, dim=-3).to(self.device)
        return rho_ux, rho_uy
    
    def get_energy_density(self, G):
        rho_E = torch.sum(G, dim=-3).to(self.device)
        return rho_E
    
    def get_macroscopic(self, F, G):
        rho = self.get_density(F)
        inv_rho = 1 / rho
        rho_ux, rho_uy = self.get_momentum(F)
        ux = rho_ux * inv_rho
        uy = rho_uy * inv_rho
        E = self.get_energy_density(G)*0.5*inv_rho
        del inv_rho, rho_ux, rho_uy
        return rho, ux, uy, E
    
    def get_w(self, T):
        one_minus_T = 1 - T
        T_term = one_minus_T * T * 0.5
        T_sq_term = T**2 * 0.25
        rest_term = one_minus_T**2
        components = [T_term, T_term, T_term, T_term, T_sq_term, T_sq_term, T_sq_term, T_sq_term, rest_term]
        del one_minus_T
        return torch.stack(components[:self.Qn], dim=-3)
    
    def get_relaxation_time(self, rho, T, F, Feq):
        tau_DL = self.muy / (rho * T) + 0.5
        diff = torch.abs(F - Feq) / Feq
        EPS = diff.mean(dim=-3)
        alpha = torch.ones_like(EPS)
        alpha = torch.where(EPS < 0.01, torch.tensor(1.0, device=EPS.device), alpha)
        alpha = torch.where(EPS < 0.1, torch.tensor(self.alpha01, device=EPS.device), alpha)
        alpha = torch.where(EPS < 1, torch.tensor(self.alpha1, device=EPS.device), alpha)
        alpha = torch.where(EPS >= 1, (1/tau_DL).clone().detach(), alpha)  
        tau_EPS = alpha * tau_DL
        tau = tau_EPS.unsqueeze(-3).expand_as(F)
        tauT = 0.5 + (tau - 0.5) / self.Pr
        omega = 1 / tau
        omegaT = 1 / tauT
        return omega, omegaT
    
    def get_Feq(self, rho, ux, uy, T):
        Feq = F_pop_torch.compute_Feq(rho, ux, self.Uax, uy, self.Uay, T, Q=self.Qn)
        return Feq
    
    def get_Geq_Newton_solver(self, rho, ux, uy, T, khi, zetax, zetay):
        # levermore_Geq_torch handles both numpy and torch inputs, and performs
        # computations on the specified device. It will return tensors since
        # the main inputs (rho, ux, uy, T) are tensors.
        Geq, khi, zetax, zetay = levermore_Geq_torch(
            self.ex, self.ey,
            ux, uy,
            T, rho,
            self.Cv, self.Qn,
            khi, zetax, zetay,
            device=self.device)
        return Geq, khi, zetax, zetay
    
    def get_maxwellian_pressure_tensor(self, rho, ux, uy, T):
        momentumx = rho * ux
        momentumy = rho * uy
        rhoT = rho * T
        P_Maxw_xx = momentumx * ux + rhoT # MB pressure tensor in xx direction
        P_Maxw_yy =momentumy* uy + rhoT 
        P_Maxw_xy = momentumx * uy 
        return P_Maxw_xx, P_Maxw_yy, P_Maxw_xy
    
    def get_pressure_tensor(self, F):
        ex2 = self.ex2.view(self._velocity_view(F.ndim))
        ey2 = self.ey2.view(self._velocity_view(F.ndim))
        exey = self.exey.view(self._velocity_view(F.ndim))
        P_xx = torch.sum(ex2 * F, dim=-3).to(self.device)
        P_yy = torch.sum(ey2 * F, dim=-3).to(self.device)
        P_xy = torch.sum(exey * F, dim=-3).to(self.device)
        del F
        return P_xx, P_yy, P_xy
    
    def get_pressure(self, T, rho):
        P = self.R*rho * T
        return P
    
    def get_qs(self, F, rho, ux, uy, T):
        P_eqxx, P_eqyy, P_eqxy = self.get_maxwellian_pressure_tensor(rho, ux, uy, T)
        P_xx, P_yy, P_xy = self.get_pressure_tensor(F)
        diff_xy = P_xy - P_eqxy
        qsx = 2 * ux * (P_xx - P_eqxx) + 2 * uy * diff_xy 
        qsy = 2 * uy * (P_yy - P_eqyy) + 2 * ux * diff_xy 
        del P_eqxx, P_eqyy, P_eqxy, P_xx, P_yy, P_xy, diff_xy
        return qsx, qsy 
    
    def from_macro_to_lattice_Gis(self,F, rho, ux, uy, T):
        w = self.get_w(T)
        qsx, qsy = self.get_qs(F, rho, ux, uy, T)
        ex = self.ex.view(self._velocity_view(w.ndim))
        ey = self.ey.view(self._velocity_view(w.ndim))
        Gis = w * (qsx.unsqueeze(-3) * ex + qsy.unsqueeze(-3) * ey) / T.unsqueeze(-3)
        del w, qsx, qsy
        return Gis
    
    def interpolate_domain(self, Fo, Go):
        # Inverse distance interpolation
        div = (1 + 2 * self.Uax)
        Fo1 = torch.zeros_like(Fo)
        Go1 = torch.zeros_like(Go)
        Fo1[..., 1:self.X] = Fo[..., 1:self.X] * (1 - self.Uax) + Fo[..., 0:self.X - 1] * self.Uax
        Go1[..., 1:self.X] = Go[..., 1:self.X] * (1 - self.Uax) + Go[..., 0:self.X - 1] * self.Uax
        Fo1[..., 0] = (Fo[..., 1] * self.Uax + Fo[..., 0] * (1 + self.Uax)) / div
        Go1[..., 0] = (Go[..., 1] * self.Uax + Go[..., 0] * (1 + self.Uax)) / div
        del div
        return Fo1, Go1
               
    def collision(self, F, G, Feq, Geq, rho, ux, uy, T ):
        omega, omegaT = self.get_relaxation_time(rho, T, F, Feq)
        Gis = self.from_macro_to_lattice_Gis(F, rho, ux, uy, T)
        F_pos_collision = F - omega * (F - Feq)
        G_pos_collision = G - omega * (G - Geq) + (omega - omegaT) * Gis
        del omega, omegaT, Gis
        return F_pos_collision, G_pos_collision
    
    def shift_operator(self, F, G):
        if F.ndim == 3:
            Fi = F[self.q_indices, self.Y_indices, self.X_indices]
            Gi = G[self.q_indices, self.Y_indices, self.X_indices]
        elif F.ndim == 4:
            Fi = F[:, self.q_indices, self.Y_indices, self.X_indices]
            Gi = G[:, self.q_indices, self.Y_indices, self.X_indices]
        else:
            raise ValueError(f"Unsupported distribution rank: {F.ndim}")
        return Fi, Gi
    
    def streaming(self, F_pos_coll, G_pos_coll):
        Fo1, Go1 = self.interpolate_domain(F_pos_coll, G_pos_coll)
        Fi, Gi = self.shift_operator(Fo1, Go1)      
        # boundary conditions
        coly = torch.arange(1, self.Y + 1, device=self.device) - 1
        Gi[..., coly, 0] = Gi[..., coly, 1]
        Gi[..., coly, self.X - 1] = Gi[..., coly, self.X - 2]
        Fi[..., coly, 0] = Fi[..., coly, 1]
        Fi[..., coly, self.X - 1] = Fi[..., coly, self.X - 2]
        del Fo1, Go1
        return Fi, Gi
    
    def case_1_initial_conditions(self):
        dtype = self.dtype
        rho0 = torch.ones((self.Y, self.X), device=self.device, dtype=dtype)  # density
        ux0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)  # fluid velocity in x
        uy0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)  # fluid velocity in y
        T0 = torch.ones((self.Y, self.X), device=self.device, dtype=dtype)  # temperature
        rho0[:, :self.Lx + 1] = 0.5
        rho0[:, self.Lx + 1:] = 2
        T0[:, :self.Lx + 1] = 0.2  # temperature
        T0[:, self.Lx + 1:] = 0.025  # temperature
        khi0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        zetax0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        zetay0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        Fi0 = self.get_Feq(rho0, ux0, uy0, T0)
        Gi0, khi, zetax, zetay = self.get_Geq_Newton_solver(rho0, ux0, uy0, T0, khi0, zetax0, zetay0)
        Fi0 = Fi0.to(self.device, dtype=dtype)
        Gi0 = Gi0.to(self.device, dtype=dtype)
        del T0
        return Fi0, Gi0, khi, zetax, zetay

    def case_2_initial_conditions(self):
        dtype = self.dtype
        rho_max = 1.0
        p_max = 0.2
        ux0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        uy0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        rho0 = torch.ones((self.Y, self.X), device=self.device, dtype=dtype)
        rho0[:, :self.Lx+1] = 1 * rho_max
        rho0[:, self.Lx+1:] = 0.125 * rho_max
        P0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        P0[:, :self.Lx+1] = 1.0 * p_max
        P0[:, self.Lx+1:] = 0.1 * p_max
        T0 = P0/(rho0*self.R)
        khi0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        zetax0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        zetay0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        Fi0 = self.get_Feq(rho0, ux0, uy0, T0)
        Gi0, khi, zetax, zetay = self.get_Geq_Newton_solver(rho0, ux0, uy0, T0, khi0, zetax0, zetay0)
        Fi0 = Fi0.to(self.device, dtype=dtype)
        Gi0 = Gi0.to(self.device, dtype=dtype)
        del P0
        return Fi0, Gi0, khi, zetax, zetay
    
    def step(self, Fi0, Gi0, khi0, zetax0, zetay0):
        # One time step: update macro, equilibrium, collision, streaming, multipliers
        rho, ux, uy, E = self.get_macroscopic(Fi0, Gi0)
        T = self.get_temp_from_energy(ux, uy, E)
        Feq = self.get_Feq(rho, ux, uy, T)
        Geq, khi, zetax, zetay = self.get_Geq_Newton_solver(rho, ux, uy, T, khi0, zetax0, zetay0)
        F_new, G_new = self.collision(Fi0, Gi0, Feq, Geq, rho, ux, uy, T)
        Fi, Gi = self.streaming(F_new, G_new)
        return Fi, Gi, khi, zetax, zetay, rho, ux, uy, T, Feq, Geq
    

def main():
    #from tqdm import tqdm
    import argparse
    import os
    import h5py
    import yaml
    from utilities import plot_simulation_results
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0,
                        help='Choose the device index (0 for cuda:0, 1 for cuda:1, 2 for cuda:2, 3 for cuda:3, -1 for cpu)')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--save', dest='save', action='store_true', help='Save file in database')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do not save file in database')
    parser.add_argument('--case', type=int, choices=[1, 2], help='Choose case 1 or 2', default=1)
    parser.add_argument('--case_config', type=str, default='Sod_cases_param.yml', help='Case parameter YAML path')
    parser.add_argument('--output_path', type=str, default=None, help='Optional output HDF5 path')
    parser.add_argument('--muy_scale', type=float, default=1.0, help='Scale factor applied to the case viscosity')
    parser.add_argument('--muy_override', type=float, default=None, help='Explicit viscosity override')
    parser.add_argument("--plot", dest='plot', action='store_true', help='Plot the results', default=False)
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile the functions', default=False)
    parser.add_argument('--newton-steps', type=int, default=15, help='Max Newton iterations for Geq solver')
    parser.add_argument('--newton-tol', type=float, default=1e-6, help='Tolerance for Newton solver')
    parser.add_argument('--precision', type=str, default='float64', choices=['float32', 'float64', 'float16', 'bfloat16'], help='Precision for simulation')
    parser.add_argument('--no-analytic-reference', dest='analytic_reference', action='store_false', help='Do not save analytic Euler reference fields')
    parser.set_defaults(save=True)
    parser.set_defaults(analytic_reference=True)


    args = parser.parse_args()
    device = get_device(args.device)

    # Set global torch dtype for precision
    if args.precision == 'float16':
        torch.set_default_dtype(torch.float16)
    elif args.precision == 'bfloat16':
        torch.set_default_dtype(torch.bfloat16)
    elif args.precision == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
        # Enable TF32 for matmul if available for better performance (only for float32)
        torch.set_float32_matmul_precision('high')

    with open(args.case_config, 'r') as f: 
        cases = yaml.load(f, Loader=yaml.FullLoader)    

    case_params = dict(cases[args.case])
    base_muy = float(case_params['muy'])
    if args.muy_override is not None:
        case_params['muy'] = float(args.muy_override)
    else:
        case_params['muy'] = base_muy * float(args.muy_scale)
    case_params['device'] = device

    print(f"Case {args.case}: SOD shock tube problem")
    print(f"Using viscosity muy={case_params['muy']} (base={base_muy}, scale={args.muy_scale})")

    sod_solver = SODSolver(
        X=case_params['X'], 
        Y=case_params['Y'], 
        Qn=case_params['Qn'], 
        alpha1=case_params['alpha1'],
        alpha01=case_params['alpha01'], 
        vuy=case_params['vuy'],
        Pr=case_params['Pr'],
        muy=case_params['muy'], 
        Uax=case_params['Uax'], 
        Uay=case_params['Uay'],
        device=case_params['device']
    )  

    if args.compile:
        print("Compiling the entire SODSolver class step method")
        # Precompile with a dummy call
        dummy_shape = (sod_solver.Qn, sod_solver.Y, sod_solver.X)
        dummy_macro_shape = (sod_solver.Y, sod_solver.X)
        dummy_F = torch.zeros(dummy_shape, device=sod_solver.device)
        dummy_G = torch.zeros(dummy_shape, device=sod_solver.device)
        dummy_khi = torch.zeros(dummy_macro_shape, device=sod_solver.device)
        dummy_zetax = torch.zeros(dummy_macro_shape, device=sod_solver.device)
        dummy_zetay = torch.zeros(dummy_macro_shape, device=sod_solver.device)
        sod_solver.is_compiled = True  # Set compilation flag
        sod_solver.step = torch.compile(sod_solver.step)
        # Precompile by running one dummy step
        with torch.no_grad():
            sod_solver.step(dummy_F, dummy_G, dummy_khi, dummy_zetax, dummy_zetay)

    initial_conditions_func = getattr(sod_solver, case_params['initial_conditions_func'])
    Fi0, Gi0, khi0, zetax0, zetay0 = initial_conditions_func()
    # --- GPU Warm-up Phase ---
    # Run a few dummy steps to warm up GPU and compile kernels before main loop
    warmup_steps = 3
    print("Warming up GPU with dummy steps...")
    with torch.no_grad():
        dummy_F = torch.zeros((sod_solver.Qn, sod_solver.Y, sod_solver.X), device=sod_solver.device)
        dummy_G = torch.zeros((sod_solver.Qn, sod_solver.Y, sod_solver.X), device=sod_solver.device)
        dummy_khi = torch.zeros((sod_solver.Y, sod_solver.X), device=sod_solver.device)
        dummy_zetax = torch.zeros((sod_solver.Y, sod_solver.X), device=sod_solver.device)
        dummy_zetay = torch.zeros((sod_solver.Y, sod_solver.X), device=sod_solver.device)
        for _ in range(warmup_steps):
            _ = sod_solver.step(dummy_F, dummy_G, dummy_khi, dummy_zetax, dummy_zetay)
    print("GPU warm-up complete.")

    analytic_rho = analytic_ux = analytic_T = analytic_P = None
    if args.analytic_reference:
        analytic_rho, analytic_ux, analytic_T, analytic_P = analytic_reference_from_case(
            args.case,
            case_params,
            steps=args.steps,
        )

    
    all_rho = []
    all_ux = []
    all_uy = []
    all_T = []
    all_Feq = []
    all_Geq = []
    all_Fi0 = []
    all_Gi0 = []
    if args.plot:
        os.makedirs('images', exist_ok=True)
    with torch.no_grad():  
        for i in range(args.steps):
            Fi0, Gi0, khi0, zetax0, zetay0, rho, ux, uy, T, Feq, Geq = sod_solver.step(Fi0, Gi0, khi0, zetax0, zetay0)
            all_rho.append(detach(rho)) 
            all_ux.append(detach(ux))
            all_uy.append(detach(uy))
            all_T.append(detach(T))
            all_Feq.append(detach(Feq))
            all_Geq.append(detach(Geq))
            all_Fi0.append(detach(Fi0))
            all_Gi0.append(detach(Gi0))
            if args.plot and (i % 100 == 0):
                P = sod_solver.get_pressure(T, rho)
                plot_simulation_results(rho, ux, T, P, i, args.case)
        if args.save:
            output_path = args.output_path or os.path.join('data_base', case_params.get('filename', f'SOD_case{args.case}.h5'))
            output_dir = os.path.dirname(output_path) or '.'
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving dataset to {output_path}")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('rho', data=all_rho) 
                f.create_dataset('ux', data=all_ux)  
                f.create_dataset('uy', data=all_uy)
                f.create_dataset('T', data=all_T)
                f.create_dataset('Feq', data=all_Feq)
                f.create_dataset('Geq', data=all_Geq)
                f.create_dataset('Fi0', data=all_Fi0)
                f.create_dataset('Gi0', data=all_Gi0) 
                if analytic_rho is not None:
                    f.create_dataset('analytic_rho', data=analytic_rho)
                    f.create_dataset('analytic_ux', data=analytic_ux)
                    f.create_dataset('analytic_T', data=analytic_T)
                    f.create_dataset('analytic_P', data=analytic_P)
                f.attrs['case'] = args.case
                f.attrs['case_config'] = os.path.abspath(args.case_config)
                f.attrs['base_muy'] = base_muy
                f.attrs['muy_scale'] = float(args.muy_scale)
                if args.muy_override is not None:
                    f.attrs['muy_override'] = float(args.muy_override)
                for key in ('X', 'Y', 'Qn', 'alpha1', 'alpha01', 'vuy', 'Pr', 'muy', 'Uax', 'Uay'):
                    f.attrs[key] = case_params[key]

if __name__=="__main__":
    main()
