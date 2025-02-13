import torch
import torch.nn as nn

class F_pop_torch(nn.Module):
    def __init__(self):
        super(F_pop_torch, self).__init__()

    @staticmethod
    def compute_Feq(rho, ux, Uax, uy, Uay, T, Q=9):
        ux_diff = ux - Uax
        uy_diff = uy - Uay

        ux_diff_sq = ux_diff ** 2
        uy_diff_sq = uy_diff ** 2

        Phi = {
            "mx": (-(ux_diff) + ux_diff_sq + T) / 2,
            "my": (-(uy_diff) + uy_diff_sq + T) / 2,
            "0x": 1 - (ux_diff_sq + T),
            "0y": 1 - (uy_diff_sq + T),
            "px": (ux_diff + ux_diff_sq + T) / 2,
            "py": (uy_diff + uy_diff_sq + T) / 2
        }


        Feq = torch.zeros((Q, T.shape[0], T.shape[1]), device=T.device)

        # Populate Feq tensor
        Feq[0] = rho * (Phi["px"] * Phi["0y"])
        Feq[1] = rho * (Phi["0x"] * Phi["py"])
        Feq[2] = rho * (Phi["mx"] * Phi["0y"])
        Feq[3] = rho * (Phi["0x"] * Phi["my"])
        Feq[4] = rho * (Phi["px"] * Phi["py"])
        Feq[5] = rho * (Phi["mx"] * Phi["py"])
        Feq[6] = rho * (Phi["mx"] * Phi["my"])
        Feq[7] = rho * (Phi["px"] * Phi["my"])
        Feq[8] = rho * (Phi["0x"] * Phi["0y"])

        return Feq

    def forward(self, rho, ux, Uax, uy, Uay, T, Q):
        return self.compute_Feq(rho, ux, Uax, uy, Uay, T, Q)
