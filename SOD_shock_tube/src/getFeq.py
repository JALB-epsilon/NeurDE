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

        components = [
            rho * (Phi["px"] * Phi["0y"]),
            rho * (Phi["0x"] * Phi["py"]),
            rho * (Phi["mx"] * Phi["0y"]),
            rho * (Phi["0x"] * Phi["my"]),
            rho * (Phi["px"] * Phi["py"]),
            rho * (Phi["mx"] * Phi["py"]),
            rho * (Phi["mx"] * Phi["my"]),
            rho * (Phi["px"] * Phi["my"]),
            rho * (Phi["0x"] * Phi["0y"]),
        ]

        return torch.stack(components[:Q], dim=-3)

    def forward(self, rho, ux, Uax, uy, Uay, T, Q):
        return self.compute_Feq(rho, ux, Uax, uy, Uay, T, Q)
