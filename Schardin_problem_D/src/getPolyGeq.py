import numpy as np
import torch.nn as nn
import torch

class Get_Geq(nn.Module):
    def __init__(self):
        super(Get_Geq, self).__init__()

    def forward(self, w, rho, E, T, qx_eq, qy_eq, Rxx_eq, Rxy_eq, Ryy_eq, ex, ey):
        ex = ex.view(9, 1, 1)
        ey = ey.view(9, 1, 1)
        
        term1 = 2. * rho * E
        term2 = (qx_eq * ex + qy_eq * ey) / T
        term3 = ((Rxx_eq - 2 * rho * E*T) * (ex**2- T) +
                 (Ryy_eq - 2 * rho * E*T) * (ey**2- T) +
                 Rxy_eq * (ex * ey)) / (2 * (T ** 2))
        
        Geq = w*(term1 + term2 + term3)
        return Geq
    

class Get_Quasi_Geq(nn.Module):
    def __init__(self):
        super(Get_Quasi_Geq, self).__init__()

    def forward(self, w, rho, ux, uy, E, T,
                Pxx_eq, Pxy_eq, Pyy_eq, Pxx, Pxy, Pyy,
                qx_eq, qy_eq, Rxx_eq, Rxy_eq, Ryy_eq, ex, ey):
        ex = ex.view(9, 1, 1)
        ey = ey.view(9, 1, 1)
        diff_Pxx = Pxx - Pxx_eq
        diff_Pxy = Pxy - Pxy_eq
        diff_Pyy = Pyy - Pyy_eq
        
        term1 = 2. * rho * E
        term2_x = ((qx_eq) +2*ux *(diff_Pxx)+2*uy *(diff_Pxy))*ex
        term2_y= ((qy_eq) +2*ux *(diff_Pxy)+2*uy *(diff_Pyy))*ey
        term3 = ((Rxx_eq - 2 * rho * E*T) * (ex**2 - T) +
                 (Ryy_eq - 2 * rho * E*T) * (ey**2 - T) +
                 Rxy_eq * (ex * ey)) / (2 * (T ** 2))
        
        Quasi_Geq = w*(term1 + (term2_x+term2_y)/T + term3)
        return Quasi_Geq