import torch
import torch.nn as nn

class F_pop_torch(nn.Module):
    def __init__(self):
        super(F_pop_torch, self).__init__()
        self.Q = 9

    @staticmethod
    def _compute_phi(ux_diff, uy_diff, T):
        ux_diff_sq = ux_diff**2
        uy_diff_sq = uy_diff**2
        return {
            "mx": (-ux_diff + ux_diff_sq + T) *0.5,
            "my": (-uy_diff + uy_diff_sq + T)  *0.5,
            "0x": 1 - (ux_diff_sq + T),
            "0y": 1 - (uy_diff_sq + T),
            "px": (ux_diff + ux_diff_sq + T)  *0.5,
            "py": (uy_diff + uy_diff_sq + T)  *0.5,
        }

    @staticmethod
    def _compute_feq_core(rho, Phi, shape, device):
        Feq = torch.zeros((9, *shape), device=device)
        Feq[0] = rho * Phi["px"] * Phi["0y"]
        Feq[1] = rho * Phi["0x"] * Phi["py"]
        Feq[2] = rho * Phi["mx"] * Phi["0y"]
        Feq[3] = rho * Phi["0x"] * Phi["my"]
        Feq[4] = rho * Phi["px"] * Phi["py"]
        Feq[5] = rho * Phi["mx"] * Phi["py"]
        Feq[6] = rho * Phi["mx"] * Phi["my"]
        Feq[7] = rho * Phi["px"] * Phi["my"]
        Feq[8] = rho * Phi["0x"] * Phi["0y"]
        return Feq

    @staticmethod
    def compute_Feq(rho, ux, Uax, uy, Uay, T):
        ux_diff = ux - Uax
        uy_diff = uy - Uay
        Phi = F_pop_torch._compute_phi(ux_diff, uy_diff, T)  # Call static method with class name
        return F_pop_torch._compute_feq_core(rho, Phi, T.shape, T.device)

    @staticmethod
    def compute_Feq_obstacle(rho, ux, Uax, uy, Uay, T, obstacle):
        ux_diff = ux[obstacle] - Uax
        uy_diff = uy[obstacle] - Uay
        Phi = F_pop_torch._compute_phi(ux_diff, uy_diff, T[obstacle])
        return F_pop_torch._compute_feq_core(rho[obstacle], Phi, (obstacle.sum(),), T.device)

    @staticmethod
    def compute_Feq_BC(rho, ux, Uax, uy, Uay, T, row, col):
        ux_diff = ux[row, col] - Uax
        uy_diff = uy[row, col] - Uay
        Phi = F_pop_torch._compute_phi(ux_diff, uy_diff, T[row, col])
        return F_pop_torch._compute_feq_core(rho[row, col], Phi, (row.shape[0],), T.device)

    def forward(self, rho, ux, Uax, uy, Uay, T, obstacle=None, bc_indices=None):
        if obstacle is None and bc_indices is None:
            return self.compute_Feq(rho, ux, Uax, uy, Uay, T)  # Call static method
        elif obstacle is not None:
            return self.compute_Feq_obstacle(rho, ux, Uax, uy, Uay, T, obstacle)  # Call static method
        elif bc_indices is not None:
            row, col = bc_indices
            return self.compute_Feq_BC(ux, uy, T, rho, Uax, Uay, row, col)  # Call static method
        else:
            obstacle_Feq = self.compute_Feq_obstacle(rho, ux, Uax, uy, Uay, T, obstacle)
            row, col = bc_indices
            bc_Feq = self.compute_Feq_BC(ux, uy, T, rho, Uax, Uay, row, col)
            return obstacle_Feq, bc_Feq