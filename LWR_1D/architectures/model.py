import torch
import torch.nn as nn


def project_scalar_conservative_mass(population, flat_macro_state):
    target_moments = flat_macro_state[:, :1]
    moment_matrix = torch.ones((1, population.shape[-1]), device=population.device, dtype=population.dtype)
    gram_inv = torch.linalg.inv(moment_matrix @ moment_matrix.transpose(0, 1))
    predicted_moments = population @ moment_matrix.transpose(0, 1)
    correction = (target_moments - predicted_moments) @ gram_inv @ moment_matrix
    return population + correction


class EquilibriumHead(nn.Module):
    def __init__(self, alpha_layer, phi_layer, activation, logit_clip=15.0, conservative_output=True):
        super().__init__()
        self.alpha = DenseNet(alpha_layer, activation)
        self.phi = DenseNet(phi_layer, activation)
        self.logit_clip = logit_clip
        self.conservative_output = bool(conservative_output)

    def forward(self, macro_state, basis):
        flat_macro_state = macro_state.movedim(1, -1).reshape(-1, macro_state.shape[1])
        alpha_coeffs = self.alpha(flat_macro_state)
        basis_coeffs = self.phi(basis)
        logits = torch.einsum("bi,ni->bn", alpha_coeffs, basis_coeffs)
        if self.logit_clip is not None:
            logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
        population = torch.exp(logits)
        if self.conservative_output:
            population = project_scalar_conservative_mass(population, flat_macro_state)
        return population


class NeurDE(nn.Module):
    def __init__(
        self,
        alpha_layer,
        phi_layer=None,
        activation="relu",
        branch_layer=None,
        learn_feq=True,
        learn_geq=False,
        logit_clip=15.0,
        conservative_output=True,
    ):
        super().__init__()
        trunk_layer = phi_layer if phi_layer is not None else branch_layer
        if trunk_layer is None:
            raise ValueError("Either phi_layer or branch_layer must be provided.")
        self.feq_head = EquilibriumHead(
            alpha_layer,
            trunk_layer,
            activation,
            logit_clip=logit_clip,
            conservative_output=conservative_output,
        ) if learn_feq else None
        self.geq_head = EquilibriumHead(
            alpha_layer,
            trunk_layer,
            activation,
            logit_clip=logit_clip,
            conservative_output=conservative_output,
        ) if learn_geq else None

    def forward(self, macro_state, basis):
        feq = self.feq_head(macro_state, basis) if self.feq_head is not None else None
        geq = self.geq_head(macro_state, basis) if self.geq_head is not None else None
        if feq is not None and geq is not None:
            return feq, geq
        if feq is not None:
            return feq
        if geq is not None:
            return geq
        raise RuntimeError("NeurDE was created without any equilibrium heads.")


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super().__init__()
        self.n_layers = len(layers) - 1
        if isinstance(nonlinearity, str):
            if nonlinearity.lower() == "tanh":
                self.nonlinearity = nn.Tanh()
            elif nonlinearity.lower() == "relu":
                self.nonlinearity = nn.ReLU()
            else:
                raise ValueError(f"{nonlinearity} is not supported")

        self.layers = nn.ModuleList()
        for idx in range(self.n_layers):
            self.layers.append(nn.Linear(layers[idx], layers[idx + 1]))
            if idx != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[idx + 1]))
                self.layers.append(self.nonlinearity)

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
