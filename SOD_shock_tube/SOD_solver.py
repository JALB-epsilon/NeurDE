import torch
import torch.nn as nn
import numpy as np
from src import F_pop_torch, levermore_Geq, levermore_Geq_torch
from utilities import detach, get_device, resolve_torch_dtype


def _as_solver_tensor(value, dtype, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


class SODSolver(nn.Module):
    def __init__(self, X=3001, Y=5, Qn=9,
                 alpha1=1.2,
                 alpha01=1.05,
                 vuy=2,
                 Pr=0.71,
                 muy=0.025,
                 Uax=0.0,
                 Uay=0.0,
                 device='cuda',
                 dtype=torch.float32):
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
        self.dtype = resolve_torch_dtype(dtype)
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
        
    def get_density(self, F): 
        pop_dim = 1 if F.dim() == 4 else 0
        rho = torch.sum(F, dim=pop_dim).to(self.device)
        return rho
    
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
        rho_E = torch.sum(G, dim=pop_dim).to(self.device)
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
        if T.dim() == 3:
            w = torch.zeros((T.shape[0], self.Qn, self.Y, self.X), device=self.device, dtype=T.dtype)
        else:
            w = torch.zeros((self.Qn, self.Y, self.X), device=self.device, dtype=T.dtype)
        one_minus_T = 1 - T
        w[..., :4, :, :] = (one_minus_T * T * 0.5).unsqueeze(-3)
        w[..., 4:8, :, :] = (T**2 * 0.25).unsqueeze(-3)
        w[..., 8, :, :] = one_minus_T**2
        del one_minus_T
        return w    
    
    def get_relaxation_time(self, rho, T, F, Feq):
        tau_DL = self.muy / (rho * T) + 0.5
        diff = torch.abs(F - Feq) / Feq
        pop_dim = 1 if F.dim() == 4 else 0
        EPS = diff.mean(dim=pop_dim)
        alpha = torch.ones_like(EPS)
        alpha = torch.where(EPS < 0.01, EPS.new_tensor(1.0), alpha)
        alpha = torch.where(EPS < 0.1, EPS.new_tensor(self.alpha01), alpha)
        alpha = torch.where(EPS < 1, EPS.new_tensor(self.alpha1), alpha)
        alpha = torch.where(EPS >= 1, 1 / tau_DL, alpha)  
        tau_EPS = alpha * tau_DL
        if F.dim() == 4:
            tau = tau_EPS.unsqueeze(1).expand(-1, self.Qn, self.Y, self.X)
        else:
            tau = tau_EPS.reshape(1, self.Y, self.X).expand(self.Qn, self.Y, self.X)
        tauT = 0.5 + (tau - 0.5) / self.Pr
        omega = 1 / tau
        omegaT = 1 / tauT
        return omega, omegaT
    
    def get_Feq(self, rho, ux, uy, T):
        Feq = F_pop_torch.compute_Feq(rho, ux, self.Uax, uy, self.Uay, T, Q=self.Qn)
        return Feq
    
    def get_Geq_Newton_solver(self, rho, ux, uy, T, khi, zetax, zetay):
        dtype = rho.dtype
        khi = _as_solver_tensor(khi, dtype, self.device)
        zetax = _as_solver_tensor(zetax, dtype, self.device)
        zetay = _as_solver_tensor(zetay, dtype, self.device)
        return levermore_Geq_torch(
            self.ex,
            self.ey,
            ux,
            uy,
            T,
            rho,
            self.Cv,
            self.Qn,
            khi,
            zetax,
            zetay,
            device=self.device,
        )
    
    def get_maxwellian_pressure_tensor(self, rho, ux, uy, T):
        momentumx = rho * ux
        momentumy = rho * uy
        rhoT = rho * T
        P_Maxw_xx = momentumx * ux + rhoT # MB pressure tensor in xx direction
        P_Maxw_yy =momentumy* uy + rhoT 
        P_Maxw_xy = momentumx * uy 
        return P_Maxw_xx, P_Maxw_yy, P_Maxw_xy
    
    def get_pressure_tensor(self, F):
        if F.dim() == 4:
            P_xx = torch.sum(F * self.ex2.view(1, self.Qn, 1, 1), dim=1).to(self.device)
            P_yy = torch.sum(F * self.ey2.view(1, self.Qn, 1, 1), dim=1).to(self.device)
            P_xy = torch.sum(F * self.exey.view(1, self.Qn, 1, 1), dim=1).to(self.device)
            del F
            return P_xx, P_yy, P_xy
        P_xx = torch.tensordot(self.ex2, F, dims=([0], [0])).to(self.device)
        P_yy = torch.tensordot(self.ey2, F, dims=([0], [0])).to(self.device)
        P_xy = torch.tensordot(self.exey, F, dims=([0], [0])).to(self.device)
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
        if F.dim() == 4:
            Gis = w * (
                qsx.unsqueeze(1) * self.ex.view(1, self.Qn, 1, 1)
                + qsy.unsqueeze(1) * self.ey.view(1, self.Qn, 1, 1)
            ) / T.unsqueeze(1)
        else:
            Gis = w * (qsx * self.ex[:, None, None] + qsy * self.ey[:, None, None]) / T[None, :, :]
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
        if F.dim() == 4:
            Fi = F[:, self.q_indices, self.Y_indices, self.X_indices]
            Gi = G[:, self.q_indices, self.Y_indices, self.X_indices]
            return Fi, Gi
        Fi = F[self.q_indices, self.Y_indices, self.X_indices]
        Gi = G[self.q_indices, self.Y_indices, self.X_indices]       
        return Fi, Gi
    
    def streaming(self, F_pos_coll, G_pos_coll):
        Fo1, Go1 = self.interpolate_domain(F_pos_coll, G_pos_coll)
        Fi, Gi = self.shift_operator(Fo1, Go1)      
        # boundary conditions
        coly = torch.arange(1, self.Y + 1, device=self.device) - 1
        if Fi.dim() == 4:
            Gi[:, :, coly, 0] = Gi[:, :, coly, 1]
            Gi[:, :, coly, self.X - 1] = Gi[:, :, coly, self.X - 2]
            Fi[:, :, coly, 0] = Fi[:, :, coly, 1]
            Fi[:, :, coly, self.X - 1] = Fi[:, :, coly, self.X - 2]
        else:
            Gi[:, coly, 0] = Gi[:, coly, 1]
            Gi[:, coly, self.X - 1] = Gi[:, coly, self.X - 2]
            Fi[:, coly, 0] = Fi[:, coly, 1]
            Fi[:, coly, self.X - 1] = Fi[:, coly, self.X - 2]
        del Fo1, Go1
        return Fi, Gi
    
    def case_1_initial_conditions(self):
        dtype = self.ex.dtype
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
        Fi0 = self.get_Feq(rho0, ux0, uy0, T0)  # F_i population
        Gi0, khi, zetax, zetay = self.get_Geq_Newton_solver(rho0, ux0, uy0, T0, khi0, zetax0, zetay0) # G_i population                                     
        Fi0 =Fi0.to(self.device)
        Gi0 = Gi0.to(self.device)
        del T0
        return Fi0, Gi0, khi, zetax, zetay
    
    def case_2_initial_conditions(self):
        rho_max = 1.0
        p_max = 0.2
        dtype = self.ex.dtype
        ux0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)  # fluid velocity in x
        uy0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)  # fluid velocity in y
        rho0 = torch.ones((self.Y, self.X), device=self.device, dtype=dtype)  # density
        rho0[:, :self.Lx+1] = 1 * rho_max
        rho0[:, self.Lx+1:] = 0.125 * rho_max
        P0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)  # pressure
        P0[:, :self.Lx+1] = 1.0 * p_max
        P0[:, self.Lx+1:] = 0.1 * p_max
        T0 = P0/(rho0*self.R)
        khi0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        zetax0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        zetay0 = torch.zeros((self.Y, self.X), device=self.device, dtype=dtype)
        Fi0 = self.get_Feq(rho0, ux0, uy0, T0)  # F_i population
        Gi0, khi, zetax, zetay = self.get_Geq_Newton_solver(rho0, ux0, uy0, T0, khi0, zetax0, zetay0) # G_i population
        Fi0 = Fi0.to(self.device)
        Gi0 = Gi0.to(self.device)
        del P0
        return Fi0, Gi0, khi, zetax, zetay
    

def main():
    from tqdm import tqdm
    import argparse
    import os
    import h5py
    import yaml
    from utilities import plot_simulation_results
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=3,
                        help='Choose the device index (0 for cpu, 1 for cuda:1, 2 for cuda:2, 3 for cuda:3)')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--save', dest='save', action='store_true', help='Save file in database')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do not save file in database')
    parser.add_argument('--case', type=int, choices=[1, 2], help='Choose case 1 or 2', default=1)
    parser.add_argument("--plot", dest='plot', action='store_true', help='Plot the results', default=False)
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile the functions', default=False)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.set_defaults(save=True)


    args = parser.parse_args()
    device = get_device(args.device)
    dtype = resolve_torch_dtype(args.dtype)

    with open("Sod_cases_param.yml", 'r') as f: 
        cases = yaml.load(f, Loader=yaml.FullLoader)    

    case_params = cases[args.case]
    case_params['device'] = device

    print(f"Case {args.case}: SOD shock tube problem")

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
                            device=case_params['device'],
                            dtype=dtype,
                            )  
    

    if args.compile:
        print("Compiling some of the functions")
        sod_solver.collision = torch.compile(sod_solver.collision, dynamic=True, fullgraph=False)
        sod_solver.streaming = torch.compile(sod_solver.streaming, dynamic=True,  fullgraph=False)  
        sod_solver.shift_operator = torch.compile(sod_solver.shift_operator,  dynamic=True, fullgraph=False)
        sod_solver.get_macroscopic = torch.compile(sod_solver.get_macroscopic, dynamic=True, fullgraph=False)
        sod_solver.get_Feq = torch.compile(sod_solver.get_Feq,  dynamic=True,  fullgraph=False)


    initial_conditions_func = getattr(sod_solver, case_params['initial_conditions_func'])
    Fi0, Gi0, khi0, zetax0, zetay0 = initial_conditions_func()

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
        for i in tqdm(range(args.steps)):
            rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
            T = sod_solver.get_temp_from_energy(ux, uy, E)
            Feq = sod_solver.get_Feq(rho, ux, uy, T)
            Geq, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver(rho, ux, uy, T, khi0, zetax0, zetay0)
            Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq, Geq, rho, ux, uy, T)
            Fi, Gi = sod_solver.streaming(Fi0, Gi0)
            all_rho.append(detach(rho)) 
            all_ux.append(detach(ux))
            all_uy.append(detach(uy))
            all_T.append(detach(T))
            all_Feq.append(detach(Feq))
            all_Geq.append(detach(Geq))
            all_Fi0.append(detach(Fi0))
            all_Gi0.append(detach(Gi0))
            Fi0 = Fi
            Gi0 = Gi
            khi0 = khi
            zetax0 = zetax
            zetay0 = zetay
           
            if args.plot and (i % 100 == 0):
                P = sod_solver.get_pressure(T, rho)
                plot_simulation_results(rho, ux, T, P, i, args.case)

        if args.save:
            os.makedirs('data_base', exist_ok=True)
            with h5py.File(f'data_base/SOD_case{args.case}.h5', 'w') as f:
                f.create_dataset('rho', data=all_rho) 
                f.create_dataset('ux', data=all_ux)  
                f.create_dataset('uy', data=all_uy)
                f.create_dataset('T', data=all_T)
                f.create_dataset('Feq', data=all_Feq)
                f.create_dataset('Geq', data=all_Geq)
                f.create_dataset('Fi0', data=all_Fi0)
                f.create_dataset('Gi0', data=all_Gi0) 

if __name__=="__main__":
    main()
