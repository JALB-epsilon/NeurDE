import torch.nn as nn
import torch     
import numpy as np  
from src import *
from utilities import detach, get_device

class Cylinder_base(nn.Module):
    def __init__(self,
                X=500,
                Y=300,
                Qn=9,
                radius=20,
                Ma0=1.7,
                Re=300,
                rho0=1,
                T0=0.2,
                alpha1=1.35,
                alpha01=1.05,
                vuy=1.4,
                Pr=0.71,
                Ns=0.6,
                device='cuda'
                ):
        super(Cylinder_base, self).__init__()
        self.X = X
        self.Y = Y
        self.CXR = round(self.X / 3)-1  # center in x for the circular cylinder
        self.CYR = round(self.Y / 2)-1  # center in y for the circular cylinder        
        self.Qn = Qn
        self.alpha1 = alpha1
        self.alpha01 = alpha01
        self.vuy = vuy
        self.Pr = Pr
        self.radius = radius 
        self.rho0 = rho0
        self.T0 = T0
        self.Ns = Ns
        self.Ma0 = Ma0
        self.Re = Re
        self.device = device
        ex_values = [1, 0, -1, 0, 1, -1, -1, 1, 0]
        ey_values = [0, 1, 0, -1, 1, 1, -1, -1, 0]
        self.get_shift_constants()
        self.ex = torch.tensor(ex_values, dtype=torch.float32, device=self.device) + self.Uax
        self.ey = torch.tensor(ey_values, dtype=torch.float32, device=self.device) + self.Uay
        self.ex1 = torch.tensor(ex_values, dtype=torch.float32, device=self.device)
        self.ey1 = torch.tensor(ey_values, dtype=torch.float32, device=self.device)
        del ex_values, ey_values
        self.get_derived_quantities()
        self.create_obstacle()

    def get_shift_constants(self):
        self.cs0 = np.sqrt(self.vuy*self.T0) # speed of sound
        self.U0 = self.Ma0 * self.cs0 #far field velocity
        #defining the shift
        self.Uax = self.U0*self.Ns
        self.Uay = 0

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


        self.muy = self.U0 * 2*self.radius / self.Re # dynamic viscosity


        # Create the column vectors for BCs
        
    def create_obstacle(self):
        # Create meshgrid directly with torch
        y, x = torch.meshgrid(torch.arange(self.Y - 1, -1, -1, device=self.device),
                            torch.arange(0, self.X, device=self.device), indexing='ij')

        # Calculate the obstacle mask
        Obs = ((x - self.CXR)**2 + (y - self.CYR)**2) < self.radius**2

        # Filter columns and rows with less than 2 True values
        Obs[:, torch.sum(Obs, dim=0) < 2] = 0
        Obs[torch.sum(Obs, dim=1) < 2, :] = 0

        # Ensure the array is a boolean tensor
        self.Obs = Obs.bool()

        # Create column vectors for boundary conditions
        self.colp = torch.arange(1, self.Y - 1, device=self.device)
        self.colx = torch.arange(self.X, device=self.device)
        self.coly = torch.arange(self.Y, device=self.device)

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
        rho = torch.sum(F, dim=0).to(self.device)
        return rho
    
    def get_momentum(self, F): 
        rho_ux = torch.tensordot(self.ex, F, dims=([0], [0])).to(self.device)
        rho_uy = torch.tensordot(self.ey, F, dims=([0], [0])).to(self.device)
        return rho_ux, rho_uy
    
    def get_energy_density(self, G):
        rho_E = torch.sum(G, dim=0).to(self.device)
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
        w = torch.zeros((self.Qn, self.Y, self.X)).to(self.device)
        one_minus_T = 1 - T
        w[:4, :, :] = one_minus_T * T *0.5
        w[4:8, :, :] = T**2 *0.25
        w[8, :, :] = one_minus_T**2
        del one_minus_T
        return w    

    def get_local_Mach(self, ux, uy, T):
        uu = self.dot_prod(ux, uy)
        cs = self.vuy * T
        return torch.sqrt(uu / cs)

    
    def get_relaxation_time(self, rho, T, F, Feq):
        tau_DL = self.muy / (rho * T) + 0.5
        diff = torch.abs(F - Feq) / Feq
        EPS = diff.mean(dim=0)
        alpha = torch.ones_like(EPS)
        alpha = torch.where(EPS < 0.01, torch.tensor(1.0, device=EPS.device), alpha)
        alpha = torch.where(EPS < 0.1, torch.tensor(self.alpha01, device=EPS.device), alpha)
        alpha = torch.where(EPS < 1, torch.tensor(self.alpha1, device=EPS.device), alpha)
        alpha = torch.where(EPS >= 1, (1/tau_DL).clone().detach(), alpha)  
        tau_EPS = alpha * tau_DL
        tau = tau_EPS.reshape(1, self.Y, self.X).expand(self.Qn, self.Y, self.X)
        tauT = 0.5 + (tau - 0.5) / self.Pr
        omega = 1 / tau
        omegaT = 1 / tauT
        return omega, omegaT
   
    def get_Feq(self, rho, ux, uy, T):
        Feq = F_pop_torch.compute_Feq(rho, ux, self.Uax, uy, self.Uay, T)
        return Feq
    
    def get_Feq_obs(self, rho, ux, uy, T):
        Feq_Obs = F_pop_torch.compute_Feq_obstacle(rho, ux, self.Uax, uy, self.Uay, T, obstacle=self.Obs)
        return Feq_Obs
    
    def get_Feq_BC(self, rho, ux, uy, T):
        Feq_BC = F_pop_torch.compute_Feq_BC(rho, ux, self.Uax, uy, self.Uay, T, self.coly, 0)        
        return Feq_BC
    
    def get_Geq_Newton_solver(self, rho, ux, uy, T, khi, zetax, zetay):
        # Convert tensors to numpy arrays
        rho_np = detach(rho) if not isinstance(rho, np.ndarray) else rho
        ux_np = detach(ux) if not isinstance(ux, np.ndarray) else ux
        uy_np = detach(uy) if not isinstance(uy, np.ndarray) else uy
        T_np = detach(T) if not isinstance(T, np.ndarray) else T
        khi = detach(khi) if not isinstance(khi, np.ndarray) else khi
        zetax = detach(zetax) if not isinstance(zetax, np.ndarray) else zetax
        zetay = detach(zetay) if not isinstance(zetay, np.ndarray) else zetay 
        # Compute Geq, khi, zetax, zetay using levermore_Geq
        Geq_np, khi, zetax, zetay = levermore_Geq(
                                                detach(self.ex), 
                                                detach(self.ey),
                                                ux_np, uy_np,
                                                 T_np, rho_np,
                                                self.Cv, self.Qn,
                                                khi, zetax, zetay
                                            ) 
        # Convert back to torch tensors
        Geq = torch.tensor(Geq_np, dtype=torch.float32,  device=self.device)
        return Geq, khi, zetax, zetay
    
    def get_Geq_Newton_solver_obs(self, rho, ux, uy, T, khi, zetax, zetay):
        # Convert tensors to numpy arrays
        rho_np = detach(rho) if not isinstance(rho, np.ndarray) else rho
        ux_np = detach(ux) if not isinstance(ux, np.ndarray) else ux
        uy_np = detach(uy) if not isinstance(uy, np.ndarray) else uy
        T_np = detach(T) if not isinstance(T, np.ndarray) else T
        khi = detach(khi) if not isinstance(khi, np.ndarray) else khi
        zetax = detach(zetax) if not isinstance(zetax, np.ndarray) else zetax
        zetay = detach(zetay) if not isinstance(zetay, np.ndarray) else zetay 
        # Compute Geq, khi, zetax, zetay using levermore_Geq
        Geq_np, khi, zetax, zetay = levermore_Geq_Obs(
                                                    detach(self.ex), 
                                                    detach(self.ey),
                                                    ux_np, uy_np,
                                                    T_np, rho_np,
                                                    self.Cv, self.Qn,
                                                    khi, zetax, zetay, 
                                                    detach(self.Obs)
                                                    ) 
        # Convert back to torch tensors
        Geq_obs = torch.tensor(Geq_np,  dtype=torch.float32, 
                               device=self.device)
        return Geq_obs, khi, zetax, zetay
    
    def get_Geq_Newton_solver_BC(self, rho, ux, uy, T, khi, zetax, zetay):
        # Convert tensors to numpy arrays
        rho_np = detach(rho) if not isinstance(rho, np.ndarray) else rho
        ux_np = detach(ux) if not isinstance(ux, np.ndarray) else ux
        uy_np = detach(uy) if not isinstance(uy, np.ndarray) else uy
        T_np = detach(T) if not isinstance(T, np.ndarray) else T
        khi = detach(khi) if not isinstance(khi, np.ndarray) else khi
        zetax = detach(zetax) if not isinstance(zetax, np.ndarray) else zetax
        zetay = detach(zetay) if not isinstance(zetay, np.ndarray) else zetay 
        # Compute Geq, khi, zetax, zetay using levermore_Geq
        Geq_np, khi, zetax, zetay = levermore_Geq_BCs(
                                                    detach(self.ex), 
                                                    detach(self.ey),
                                                    ux_np, uy_np,
                                                    T_np, rho_np,
                                                    self.Cv, self.Qn,
                                                    khi, zetax, zetay, detach(self.coly), 0
                                                    ) 
        # Convert back to torch tensors
        Geq_BC = torch.tensor(Geq_np, dtype=torch.float32, 
                              device=self.device)
        return Geq_BC, khi, zetax, zetay


    def get_obs_distribution(self, rho, ux, uy, T,  khi, zetax, zetay):
        ux_obs = torch.where(self.Obs, torch.tensor(0.0, device=self.device), ux)
        uy_obs = torch.where(self.Obs, torch.tensor(0.0, device=self.device), uy)
        T_obs = torch.where(self.Obs, torch.tensor(self.T0, device=self.device), T)
        rho_obs = torch.where(self.Obs, torch.tensor(1.0, device=self.device), rho)
                    
        #ux_obs = ux.clone()
        #uy_obs = uy.clone()
        #T_obs = T.clone()
        #rho_obs = rho.clone()  
        #ux_obs[self.Obs] = 0
        #uy_obs[self.Obs] = 0
        #T_obs[self.Obs] = self.T0
        #rho_obs[self.Obs] = 1
          
        Fi_obs_cyl = self.get_Feq_obs(rho_obs, ux_obs, uy_obs, T_obs)

        Gi_obs_cyl,khi_obs, zetax_obs, zetay_obs  = self.get_Geq_Newton_solver_obs(rho_obs,
                                                            ux_obs,
                                                            uy_obs,
                                                            T_obs,
                                                            khi,
                                                            zetax,
                                                            zetay)                                                                                  
  
        # Inlet
        ux_obs[self.coly, 0] = self.U0
        uy_obs[self.coly, 0] = 0
        T_obs[self.coly, 0] = self.T0
        rho_obs[self.coly, 0] = self.rho0


        Fi_obs_Inlet = self.get_Feq_BC(rho_obs, ux_obs, uy_obs, T_obs)
        Gi_obs_Inlet, khi, zetax, zetay = self.get_Geq_Newton_solver_BC(rho_obs,
                                                            ux_obs,
                                                            uy_obs,
                                                            T_obs,
                                                            khi_obs,
                                                            zetax_obs,
                                                            zetay_obs)
  

        Gi_obs_Inlet = Gi_obs_Inlet.to(self.device) 
        Gi_obs_cyl = Gi_obs_cyl.to(self.device)

        return Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet
    
    def get_maxwellian_pressure_tensor(self, rho, ux, uy, T):
        momentumx = rho * ux
        momentumy = rho * uy
        rhoT = rho * T
        P_Maxw_xx = momentumx * ux + rhoT # pressure tensor in xx in equilibrium
        P_Maxw_yy =momentumy* uy + rhoT  # pressure tensor in yy in equilibrium
        P_Maxw_xy = momentumx * uy  # pressure tensor in xy in equilibrium
        return P_Maxw_xx, P_Maxw_yy, P_Maxw_xy
    
    def get_pressure_tensor(self, F):
        P_xx = torch.tensordot(self.ex2, F, dims=([0], [0])).to(self.device)
        P_yy = torch.tensordot(self.ey2, F, dims=([0], [0])).to(self.device)
        P_xy = torch.tensordot(self.exey, F, dims=([0], [0])).to(self.device)
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
        return qsx, qsy 
    
    def from_macro_to_lattice_Gis(self,F, rho, ux, uy, T):
        w = self.get_w(T)
        qsx, qsy = self.get_qs(F, rho, ux, uy, T)
        Gis = w * (qsx * self.ex[:, None, None] + qsy * self.ey[:, None, None]) / T[None, :, :]
        return Gis
    
    def interpolate_domain(self, Fo, Go):
        # Inverse distance interpolation
        div = (1 + 2 * self.Uax)
        Fo1 = torch.zeros((self.Qn, self.Y, self.X)).to(self.device)
        Go1 = torch.zeros((self.Qn, self.Y, self.X)).to(self.device)
        Fo1[:, :, 1:self.X] = Fo[:, :, 1:self.X] * (1 - self.Uax) + Fo[:, :, 0:self.X - 1] * self.Uax
        Go1[:, :, 1:self.X] = Go[:, :, 1:self.X] * (1 - self.Uax) + Go[:, :, 0:self.X - 1] * self.Uax
        Fo1[:, :, 0] = (Fo[:, :, 1] * self.Uax + Fo[:, :, 0] * (1 + self.Uax)) / div
        Go1[:, :, 0] = (Go[:, :, 1] * self.Uax + Go[:, :, 0] * (1 + self.Uax)) / div
        return Fo1, Go1
               
    def collision(self, F, G, Feq, Geq, rho, ux, uy, T ):
        omega, omegaT = self.get_relaxation_time(rho, T, F, Feq)
        Gis = self.from_macro_to_lattice_Gis(F, rho, ux, uy, T)
        F_pos_collision = F - omega * (F - Feq)
        G_pos_collision = G - omega * (G - Geq) + (omega - omegaT) * Gis
        return F_pos_collision, G_pos_collision
    
    def shift_operator(self, F, G):
        Fi = F[self.q_indices, self.Y_indices, self.X_indices]
        Gi = G[self.q_indices, self.Y_indices, self.X_indices]
        return Fi, Gi
    
    
    def streaming(self, F_pos_coll, G_pos_coll):
        Fo1, Go1 = self.interpolate_domain(F_pos_coll, G_pos_coll)
        Fi, Gi = self.shift_operator(Fo1, Go1)      
        return Fi, Gi
    
    def initial_conditions(self):
        # Initial condition
        rho = torch.ones((self.Y, self.X))
        ux =  torch.full((self.Y, self.X), self.U0)
        uy = torch.zeros((self.Y, self.X))
        T = torch.full((self.Y, self.X), self.T0)

        khi0 = np.zeros((self.Y, self.X))
        zetax0 = np.zeros((self.Y, self.X))
        zetay0 = np.zeros((self.Y, self.X))

        Fi0 = self.get_Feq(rho, ux, uy, T)
 
        Gi0, khi, zetax, zetay = self.get_Geq_Newton_solver(rho,
                                                            ux, 
                                                            uy, 
                                                            T,
                                                            khi0,
                                                            zetax0, 
                                                            zetay0) 
        Fi0 = Fi0.to(self.device)
        Gi0 = Gi0.to(self.device)
        del rho, ux, uy, T, khi0, zetax0, zetay0
        return Fi0, Gi0, khi, zetax, zetay
                                              

    def enforce_Obs_and_BC(self, Fi, Gi, Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet):

        Fi_obs = Fi.clone()
        Gi_obs = Gi.clone()

        # Obstacle
        Fi_obs[:, self.Obs] = Fi_obs_cyl
        Gi_obs[:, self.Obs] = Gi_obs_cyl

        #Inlet
        Fi_obs[:, self.coly, 0] = Fi_obs_Inlet
        Gi_obs[:, self.coly, 0] = Gi_obs_Inlet

        # Outlet
        Fi_obs[:, self.coly, self.X-1] = Fi_obs[:, self.coly, self.X-2]
        Gi_obs[:, self.coly, self.X-1] = Gi_obs[:, self.coly, self.X-2]

        # Upper wall
        Fi_obs[:, 0, self.colx] = Fi_obs[:, 1, self.colx]
        Gi_obs[:, 0, self.colx] = Gi_obs[:, 1, self.colx]

        # Lower wall
        Fi_obs[:, self.Y-1, self.colx] = Fi_obs[:, self.Y-2, self.colx]
        Gi_obs[:, self.Y-1, self.colx] = Gi_obs[:, self.Y-2, self.colx]


        return Fi_obs, Gi_obs

def main():
    from tqdm import tqdm
    import argparse
    import os
    import h5py
    import yaml
    from utilities import plot_simulation_results
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=3,
                        help='Choose the device index (0 for cpu, 1 for cuda:1, 2 for cuda:2, 3 for cuda:3)')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--save', dest='save', action='store_true', help='Save file in database')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do not save file in database')
    parser.add_argument("--plot", dest='plot', action='store_true', help='Plot the results', default=False)
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile the functions', default=False)
    parser.set_defaults(save=True)

    args = parser.parse_args()
    device = get_device(args.device)
    with open ('cylinder_param.yml', 'r') as file:
        config = yaml.safe_load(file)

    config['device'] = device
    os.makedirs('images', exist_ok=True)    
    print(f"Cylinder case parameters {config}")
    cylinder_solver = Cylinder_base(
                                    X=config['X'],
                                    Y=config['Y'],
                                    Qn=config['Qn'],
                                    radius=config['radius'],
                                    Ma0=config['Ma0'],
                                    Re=config['Re'],
                                    rho0=config['rho0'],
                                    T0=config['T0'],
                                    alpha1=config['alpha1'],
                                    alpha01=config['alpha01'],
                                    vuy=config['vuy'],
                                    Pr=config['Pr'],
                                    Ns=config['Ns'],
                                    device=device
                                    )
    
   
    if args.compile:    
        cylinder_solver.collision = torch.compile(cylinder_solver.collision, dynamic=True, fullgraph=False)
        cylinder_solver.streaming = torch.compile(cylinder_solver.streaming, dynamic=True, fullgraph=False)
        cylinder_solver.shift_operator = torch.compile(cylinder_solver.shift_operator, dynamic=True, fullgraph=False)
        cylinder_solver.get_macroscopic = torch.compile(cylinder_solver.get_macroscopic, dynamic=True, fullgraph=False)
        cylinder_solver.get_Feq = torch.compile(cylinder_solver.get_Feq, dynamic=True, fullgraph=False) 
        cylinder_solver.get_relaxation_time = torch.compile(cylinder_solver.get_relaxation_time, dynamic=True, fullgraph=False)
      
    Fi0, Gi0, khi0, zetax0, zetay0 = cylinder_solver.initial_conditions()

    all_rho = []
    all_ux = []
    all_uy = []
    all_T = []
    all_Feq = []
    all_Geq = []
    all_Fi0 = []
    all_Gi0 = []
    all_Fi_obs_cyl = []
    all_Gi_obs_cyl = []
    all_Fi_obs_Inlet = []
    all_Gi_obs_Inlet = []


    if args.plot:
        os.makedirs('images', exist_ok=True)
    with torch.no_grad():  
        for i in tqdm(range(args.steps)):
            all_Fi0.append(detach(Fi0))
            all_Gi0.append(detach(Gi0))
            rho, ux, uy, E = cylinder_solver.get_macroscopic(Fi0, Gi0)
            T = cylinder_solver.get_temp_from_energy(ux, uy, E)

            all_rho.append(detach(rho)) 
            all_ux.append(detach(ux))
            all_uy.append(detach(uy))
            all_T.append(detach(T))



            Feq = cylinder_solver.get_Feq(rho, ux, uy, T)
            Geq, khi, zetax, zetay = cylinder_solver.get_Geq_Newton_solver(rho,
                                                                           ux, 
                                                                           uy, 
                                                                           T, 
                                                                           khi0, 
                                                                           zetax0, 
                                                                           zetay0)
            
            all_Feq.append(detach(Feq))
            all_Geq.append(detach(Geq))

            Fi0, Gi0 = cylinder_solver.collision(Fi0, Gi0, Feq, Geq, rho, ux, uy, T)
            Fi, Gi = cylinder_solver.streaming(Fi0, Gi0)
            Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet = cylinder_solver.get_obs_distribution(
                                                                            rho,
                                                                            ux, 
                                                                            uy,
                                                                            T,
                                                                            khi,
                                                                            zetax,
                                                                            zetay)
            
            all_Fi_obs_cyl.append(detach(Fi_obs_cyl))
            all_Gi_obs_cyl.append(detach(Gi_obs_cyl))
            all_Fi_obs_Inlet.append(detach(Fi_obs_Inlet))
            all_Gi_obs_Inlet.append(detach(Gi_obs_Inlet))
            
            Fi_new, Gi_new = cylinder_solver.enforce_Obs_and_BC(Fi,
                                                                Gi,
                                                                Fi_obs_cyl,
                                                                Gi_obs_cyl,
                                                                Fi_obs_Inlet,
                                                                Gi_obs_Inlet)
            

    
            # Update the distributions
            Fi0 = Fi_new
            Gi0 = Gi_new
            khi0 = khi
            zetax0 = zetax
            zetay0 = zetay

            if args.plot and (i % 100 == 0):
                Ma = cylinder_solver.get_local_Mach(ux, uy, T)
                plot_simulation_results(detach(Ma), i)
                
            

        if args.save:
            os.makedirs('data_base', exist_ok=True)
            with h5py.File(f'data_base/cylinder_case.h5', 'w') as f:
                f.create_dataset('rho', data=all_rho) 
                f.create_dataset('ux', data=all_ux)  
                f.create_dataset('uy', data=all_uy)
                f.create_dataset('T', data=all_T)
                f.create_dataset('Feq', data=all_Feq)
                f.create_dataset('Geq', data=all_Geq)
                f.create_dataset('Fi0', data=all_Fi0)
                f.create_dataset('Gi0', data=all_Gi0) 
                f.create_dataset('Fi_obs_cyl', data=all_Fi_obs_cyl)
                f.create_dataset('Gi_obs_cyl', data=all_Gi_obs_cyl)
                f.create_dataset('Fi_obs_Inlet', data=all_Fi_obs_Inlet)
                f.create_dataset('Gi_obs_Inlet', data=all_Gi_obs_Inlet)

        
if __name__=="__main__":
    main()       