import math
import torch
import torch.nn as nn


def _resolve_nullspace_gamma(nullspace_gamma, reference_tensor):
    if torch.is_tensor(nullspace_gamma):
        gamma = nullspace_gamma.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
    else:
        gamma = torch.tensor(float(nullspace_gamma), device=reference_tensor.device, dtype=reference_tensor.dtype)
    return gamma.clamp(0.0, 1.0)


def project_conserved_moments(flat_population, flat_macro_state, basis, nullspace_gamma=1.0):
    basis = basis.to(device=flat_population.device, dtype=flat_population.dtype)
    ex = basis[:, 0]
    ey = basis[:, 1]
    ones = torch.ones_like(ex)

    moment_basis = torch.stack([ones, ex, ey], dim=0)
    gram = moment_basis @ moment_basis.transpose(0, 1)
    gram_inv = torch.linalg.inv(gram)

    rho = flat_macro_state[:, 0]
    ux = flat_macro_state[:, 1]
    uy = flat_macro_state[:, 2]

    target_moments = torch.stack([rho, rho * ux, rho * uy], dim=-1)
    predicted_moments = torch.stack(
        [
            flat_population.sum(dim=-1),
            flat_population @ ex,
            flat_population @ ey,
        ],
        dim=-1,
    )
    raw_row_coeffs = predicted_moments @ gram_inv.transpose(0, 1)
    raw_row_component = raw_row_coeffs @ moment_basis
    target_row_coeffs = target_moments @ gram_inv.transpose(0, 1)
    target_row_component = target_row_coeffs @ moment_basis
    nullspace_component = flat_population - raw_row_component
    gamma = _resolve_nullspace_gamma(nullspace_gamma, flat_population)
    return target_row_component + gamma * nullspace_component


def project_energy_moments(flat_population, flat_macro_state, basis, cv, nullspace_gamma=1.0, positivity_eps=1.0e-8):
    del basis
    ones = torch.ones((1, flat_population.shape[-1]), device=flat_population.device, dtype=flat_population.dtype)
    gram = ones @ ones.transpose(0, 1)
    gram_inv = torch.linalg.inv(gram)

    rho = flat_macro_state[:, 0]
    ux = flat_macro_state[:, 1]
    uy = flat_macro_state[:, 2]
    T = flat_macro_state[:, 3]
    E = cv * T + 0.5 * (ux.square() + uy.square())

    target_moments = (2.0 * rho * E).unsqueeze(-1)
    predicted_moments = flat_population.sum(dim=-1, keepdim=True)
    raw_row_coeffs = predicted_moments @ gram_inv.transpose(0, 1)
    raw_row_component = raw_row_coeffs @ ones
    target_row_coeffs = target_moments @ gram_inv.transpose(0, 1)
    target_row_component = target_row_coeffs @ ones
    nullspace_component = flat_population - raw_row_component
    gamma = _resolve_nullspace_gamma(nullspace_gamma, flat_population)
    projected = target_row_component + gamma * nullspace_component
    if positivity_eps is not None:
        eps = torch.tensor(float(positivity_eps), device=projected.device, dtype=projected.dtype)
        projected = projected.clamp_min(eps)
        projected_sum = projected.sum(dim=-1, keepdim=True).clamp_min(eps)
        projected = projected * (target_moments / projected_sum)
    return projected


def bounded_residual_population(base_population, predicted_population, residual_scale, eps=1.0e-12):
    base_population = torch.clamp(base_population, min=eps)
    predicted_population = torch.clamp(predicted_population, min=eps)
    log_ratio = torch.log(predicted_population / base_population)
    return base_population * torch.exp(residual_scale * torch.tanh(log_ratio))


def enforce_sod_1d_symmetry(flat_population):
    symmetric_population = flat_population.clone()
    pair_13 = 0.5 * (flat_population[:, 1] + flat_population[:, 3])
    pair_47 = 0.5 * (flat_population[:, 4] + flat_population[:, 7])
    pair_56 = 0.5 * (flat_population[:, 5] + flat_population[:, 6])
    symmetric_population[:, 1] = pair_13
    symmetric_population[:, 3] = pair_13
    symmetric_population[:, 4] = pair_47
    symmetric_population[:, 7] = pair_47
    symmetric_population[:, 5] = pair_56
    symmetric_population[:, 6] = pair_56
    return symmetric_population


class EquilibriumHead(nn.Module):
    def __init__(self, alpha_layer, phi_layer, activation, logit_clip=15.0, hard_match_moments=False):
        super().__init__()
        self.alpha = DenseNet(alpha_layer, activation)
        self.phi = DenseNet(phi_layer, activation)
        self.logit_clip = logit_clip
        self.hard_match_moments = hard_match_moments

    @staticmethod
    def project_conserved_moments(flat_population, flat_macro_state, basis, nullspace_gamma=1.0):
        return project_conserved_moments(flat_population, flat_macro_state, basis, nullspace_gamma=nullspace_gamma)

    def forward(self, macro_state, basis):
        flat_macro_state = macro_state.movedim(1, -1).reshape(-1, macro_state.shape[1])
        alpha_coeffs = self.alpha(flat_macro_state)
        basis_coeffs = self.phi(basis)
        logits = torch.einsum("bi,ni->bn", alpha_coeffs, basis_coeffs)
        if self.logit_clip is not None:
            logits = logits.clamp(min=-self.logit_clip, max=self.logit_clip)
        population = torch.exp(logits)
        if self.hard_match_moments:
            population = self.project_conserved_moments(population, flat_macro_state, basis)
        return population


class NeurDE(nn.Module):
    def __init__(
        self,
        alpha_layer,
        phi_layer=None,
        activation="relu",
        branch_layer=None,
        learn_feq=True,
        learn_geq=True,
        logit_clip=15.0,
        project_feq_moments=False,
        enforce_sod_symmetry=False,
        feq_nullspace_gamma=1.0,
        feq_nullspace_gamma_mode="fixed",
        feq_nullspace_gamma_min=0.6,
        feq_nullspace_gamma_max=1.0,
        geq_mode="learned",
        geq_cv=1.0,
        geq_nullspace_gamma=1.0,
        geq_nullspace_gamma_mode="fixed",
        geq_nullspace_gamma_min=0.6,
        geq_nullspace_gamma_max=1.0,
    ):
        super(NeurDE, self).__init__()
        trunk_layer = phi_layer if phi_layer is not None else branch_layer
        if trunk_layer is None:
            raise ValueError("Either phi_layer or branch_layer must be provided.")
        feq_nullspace_gamma_mode = str(feq_nullspace_gamma_mode).lower()
        if feq_nullspace_gamma_mode not in {"fixed", "learned_global"}:
            raise ValueError(f"Unsupported feq_nullspace_gamma_mode: {feq_nullspace_gamma_mode}")
        geq_mode = str(geq_mode).lower()
        if geq_mode not in {"learned", "conservative"}:
            raise ValueError(f"Unsupported geq_mode: {geq_mode}")
        geq_nullspace_gamma_mode = str(geq_nullspace_gamma_mode).lower()
        if geq_nullspace_gamma_mode not in {"fixed", "learned_global"}:
            raise ValueError(f"Unsupported geq_nullspace_gamma_mode: {geq_nullspace_gamma_mode}")
        self.enforce_sod_symmetry = bool(enforce_sod_symmetry)
        self.project_feq_moments = bool(project_feq_moments)
        self.feq_nullspace_gamma_mode = feq_nullspace_gamma_mode
        self.feq_nullspace_gamma_min = float(feq_nullspace_gamma_min)
        self.feq_nullspace_gamma_max = float(feq_nullspace_gamma_max)
        if self.feq_nullspace_gamma_max <= self.feq_nullspace_gamma_min:
            raise ValueError("feq_nullspace_gamma_max must be greater than feq_nullspace_gamma_min.")
        if self.feq_nullspace_gamma_mode == "learned_global":
            init_gamma = min(max(float(feq_nullspace_gamma), self.feq_nullspace_gamma_min + 1.0e-6), self.feq_nullspace_gamma_max - 1.0e-6)
            normalized = (init_gamma - self.feq_nullspace_gamma_min) / (self.feq_nullspace_gamma_max - self.feq_nullspace_gamma_min)
            gamma_logit = math.log(normalized / (1.0 - normalized))
            self.feq_nullspace_gamma_logit = nn.Parameter(torch.tensor(gamma_logit, dtype=torch.float32))
            self.feq_nullspace_gamma = None
        else:
            self.feq_nullspace_gamma = float(feq_nullspace_gamma)
            self.register_parameter("feq_nullspace_gamma_logit", None)
        self.geq_mode = geq_mode
        self.geq_cv = float(geq_cv)
        self.geq_nullspace_gamma_mode = geq_nullspace_gamma_mode
        self.geq_nullspace_gamma_min = float(geq_nullspace_gamma_min)
        self.geq_nullspace_gamma_max = float(geq_nullspace_gamma_max)
        if self.geq_nullspace_gamma_max <= self.geq_nullspace_gamma_min:
            raise ValueError("geq_nullspace_gamma_max must be greater than geq_nullspace_gamma_min.")
        if self.geq_nullspace_gamma_mode == "learned_global":
            init_gamma = min(max(float(geq_nullspace_gamma), self.geq_nullspace_gamma_min + 1.0e-6), self.geq_nullspace_gamma_max - 1.0e-6)
            normalized = (init_gamma - self.geq_nullspace_gamma_min) / (self.geq_nullspace_gamma_max - self.geq_nullspace_gamma_min)
            gamma_logit = math.log(normalized / (1.0 - normalized))
            self.geq_nullspace_gamma_logit = nn.Parameter(torch.tensor(gamma_logit, dtype=torch.float32))
            self.geq_nullspace_gamma = None
        else:
            self.geq_nullspace_gamma = float(geq_nullspace_gamma)
            self.register_parameter("geq_nullspace_gamma_logit", None)
        self.feq_head = EquilibriumHead(
            alpha_layer,
            trunk_layer,
            activation,
            logit_clip=logit_clip,
            hard_match_moments=False,
        ) if learn_feq else None
        self.geq_head = EquilibriumHead(alpha_layer, trunk_layer, activation, logit_clip=logit_clip) if learn_geq else None

    def get_feq_nullspace_gamma(self, reference_tensor):
        if self.feq_nullspace_gamma_mode == "learned_global":
            span = self.feq_nullspace_gamma_max - self.feq_nullspace_gamma_min
            gamma = self.feq_nullspace_gamma_min + span * torch.sigmoid(self.feq_nullspace_gamma_logit)
            return gamma.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        return torch.tensor(
            self.feq_nullspace_gamma,
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        ).clamp(0.0, 1.0)

    def get_geq_nullspace_gamma(self, reference_tensor):
        if self.geq_nullspace_gamma_mode == "learned_global":
            span = self.geq_nullspace_gamma_max - self.geq_nullspace_gamma_min
            gamma = self.geq_nullspace_gamma_min + span * torch.sigmoid(self.geq_nullspace_gamma_logit)
            return gamma.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        return torch.tensor(
            self.geq_nullspace_gamma,
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        ).clamp(0.0, 1.0)

    def forward(self, macro_state, basis):
        feq = self.feq_head(macro_state, basis) if self.feq_head is not None else None
        geq = self.geq_head(macro_state, basis) if self.geq_head is not None else None
        if self.enforce_sod_symmetry:
            if feq is not None:
                feq = enforce_sod_1d_symmetry(feq)
            if geq is not None:
                geq = enforce_sod_1d_symmetry(geq)
        flat_macro_state = None
        if feq is not None and self.project_feq_moments:
            flat_macro_state = macro_state.movedim(1, -1).reshape(-1, macro_state.shape[1])
            feq_nullspace_gamma = self.get_feq_nullspace_gamma(feq)
            feq = project_conserved_moments(
                feq,
                flat_macro_state,
                basis,
                nullspace_gamma=feq_nullspace_gamma,
            )
        if geq is not None and self.geq_mode == "conservative":
            if flat_macro_state is None:
                flat_macro_state = macro_state.movedim(1, -1).reshape(-1, macro_state.shape[1])
            geq_nullspace_gamma = self.get_geq_nullspace_gamma(geq)
            geq = project_energy_moments(
                geq,
                flat_macro_state,
                basis,
                self.geq_cv,
                nullspace_gamma=geq_nullspace_gamma,
            )
        if feq is not None and geq is not None:
            return feq, geq
        if feq is not None:
            return feq
        if geq is not None:
            return geq
        raise RuntimeError("NeurDE was created without any equilibrium heads.")

class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        if isinstance(nonlinearity, str):
            if nonlinearity.lower() == 'tanh':
                self.nonlinearity = nn.Tanh()
            elif nonlinearity.lower() == 'relu':
                self.nonlinearity = nn.ReLU()
            else:
                raise ValueError(f'{nonlinearity} type {type(nonlinearity)} is not supported')

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))
                self.layers.append(self.nonlinearity)

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    
if __name__ == "__main__":
    # Define the model
    alpha_layer = [4, 50, 50, 50, 50]
    phi_layer = [2, 50, 50, 50, 50]
    activation = 'relu'
    model = NeurDE(alpha_layer, phi_layer, activation)
    print(model)
